# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import operator

# import the custom logging module
from ._logging import get_logger

# make a logger for this file
logger = get_logger(__name__)

at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch


def emb_ops_horizontal_fusion(fx_g):
    logger.info("Fusing horizontal parallel ops.")
    zentorch_embed_ops_dict = {
        zt_ops.zendnn_embedding_bag.default:
            zt_ops.zendnn_horizontal_embedding_bag_group.default,
        zt_ops.zendnn_embedding.default:
            zt_ops.zendnn_horizontal_embedding_group.default,
    }
    groups = {}

    for node in fx_g.graph.nodes:
        if node.target in zentorch_embed_ops_dict.keys():
            users = list(node.users.keys())
            user_node = None

            if node.target == zt_ops.zendnn_embedding.default:
                if len(users) == 1:
                    user_node = users[0]
            elif node.target == zt_ops.zendnn_embedding_bag.default:
                for user in users:
                    if user_node is None and len(user.users.keys()) == 1:
                        user_node = user
                    elif user_node is not None and len(user.users.keys()) == 1:
                        user_node = None
                        break

            if user_node is not None:

                if node.target == zt_ops.zendnn_embedding.default:
                    common_output_node = user_node
                    node_name = common_output_node.name
                    if node_name in groups:
                        if groups[node_name]["type"] == "embedding_bag":
                            logger.info(
                                "Cannot fuse embedding bag and embedding with \
                                 common node. This is because of the function \
                                 prototype difference between the \
                                 aten.embedding and aten.embeddingbag ops and \
                                 their corresponding zentorch group ops."
                            )
                            return fx_g
                        groups[node_name]["nodes"].append(node)
                    else:
                        groups[node_name] = {
                            "type": "embedding",
                            "nodes": [node],
                        }
                elif node.target == zt_ops.zendnn_embedding_bag.default:
                    common_output_node = list(user_node.users.keys())[0]
                    node_name = common_output_node.name
                    if node_name in groups:
                        if groups[node_name]["type"] == "embedding":
                            logger.info(
                                "Cannot fuse embedding bag and embedding with \
                                 common node. This is because of the function \
                                 prototype difference between the \
                                 aten.embedding and aten.embeddingbag ops and \
                                 their corresponding zentorch group ops."
                            )
                            return fx_g
                        groups[node_name]["nodes"].append(node)
                    else:
                        groups[node_name] = {
                            "type": "embedding_bag",
                            "nodes": [node],
                        }

    def populate_default_args(list_new_args, type_of_node):
        # Depending on the already existing values of arguments in
        # list_new_args, the remaining arguments are populated for embedding
        # ops. The type_of_node decides whether the arguments that are
        # populated are of embedding op or embeddingbag op. Embedding requires
        # two out of 7 arguments mandatorily and other arguments can have
        # either default or user specified values. Similarly, EmbeddingBag
        # requires three out of nine arguments mandatorily and other arguments
        # can either have default or user specified values.
        # The list_new_args list is populated as specified above and returned.
        num_ops = len(list_new_args[0])
        default_args = None
        if type_of_node == "embedding":
            default_args = [None, None, -1, 0, 0]
        elif type_of_node == "embedding_bag":
            default_args = [None, None, None, 0, 0, 0, None, 0, -1]

        non_empty_args_idx = 0
        for idx, l in enumerate(list_new_args):
            if len(l) == 0:
                non_empty_args_idx = idx
                break

        for idx in range(non_empty_args_idx, len(default_args)):
            list_new_args[idx] = [default_args[idx] for _ in range(num_ops)]

        return list_new_args

    for group in groups:
        if len(groups[group]["nodes"]) > 1:
            op_count = 0
            # embedding_bag has more parameters than embedding. So creating new
            # args list with max of the number of parameters of both ops, which
            # is 9. Empty lists are further removed.
            list_new_args = [[] for _ in range(9)]
            last_node = groups[group]["nodes"][-1]
            traversed_nodes = set()

            for node in groups[group]["nodes"]:
                node_args_len = len(node.args)
                for i in range(node_args_len):
                    if node.args[i] is False:
                        list_new_args[i].append(0)
                    elif node.args[i] is True:
                        list_new_args[i].append(1)
                    else:
                        list_new_args[i].append(node.args[i])
            if node.target == zt_ops.zendnn_embedding.default:
                list_new_args = populate_default_args(
                    list_new_args, type_of_node="embedding"
                )
            elif node.target == zt_ops.zendnn_embedding_bag.default:
                list_new_args = populate_default_args(
                    list_new_args, type_of_node="embedding_bag"
                )

            for node in groups[group]["nodes"]:
                if node.target == zt_ops.zendnn_embedding_bag.default:
                    total_prev_outputs_emb_bag = op_count * 4
                    for temp_node in list(node.users.keys()):
                        if (
                            temp_node.target == operator.getitem
                            and temp_node not in traversed_nodes
                        ):
                            temp_node_args = (
                                temp_node.args[0],
                                temp_node.args[1] + total_prev_outputs_emb_bag,
                            )
                            temp_node.args = temp_node_args
                        last_node.append(temp_node)
                        traversed_nodes.add(temp_node)

                if node != last_node:
                    node.replace_all_uses_with(last_node)
                    fx_g.graph.erase_node(node)

                op_count += 1

            if op_count > 1:
                idx = -1
                while len(list_new_args[idx]) == 0:
                    list_new_args.pop()
                last_node.args = tuple(list_new_args)
                if last_node.target in zentorch_embed_ops_dict.keys():
                    last_node.target = zentorch_embed_ops_dict[last_node.target]
            elif op_count == 1:
                last_node.target = node.target

            if node.target == zt_ops.zendnn_horizontal_embedding_group.default:
                common_output_node = list(last_node.users.keys())[0]
                getitem_nodes = []
                for getitem_num in range(op_count):
                    new_node = fx_g.graph.create_node(
                        op="call_function",
                        target=operator.getitem,
                        args=(last_node, getitem_num),
                    )
                    last_node.append(new_node)
                    getitem_nodes.append(new_node)
                if len(common_output_node.args) == 2:
                    common_output_node.args = (
                        getitem_nodes,
                        common_output_node.args[1],
                    )
                else:
                    common_output_node.args = (getitem_nodes,)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    return fx_g


def vertical_mlp_fusion(fx_graph):
    def return_next_addmm(users):
        # GroupMLP fusion is possible only when the consecutive Linear layers
        # have only one output, which must go to the next Linear layer. These
        # two conditions translate to checking if length of users is 1 and the
        # the user is either zendnn_addmm or zendnn_addmm_1dbias. Only when
        # these two conditions are true, True and the next Linear layer node
        # is returned.
        if len(users) == 1:
            if users[0].target in (
                zt_ops.zendnn_addmm.default,
                zt_ops.zendnn_addmm_1dbias.default,
            ):
                return users[0]

        return None

    for node in fx_graph.graph.nodes:
        if node.target == at_ops.detach.default:
            fx_graph.graph.erase_node(node)

    addmm_groups = [[]]
    nodes_traversed = set()
    for node in fx_graph.graph.nodes:
        if node.target in (
            zt_ops.zendnn_addmm.default,
            zt_ops.zendnn_addmm_1dbias.default,
        ):
            while node not in nodes_traversed:
                addmm_groups[-1].append(node)
                nodes_traversed.add(node)
                result_node = return_next_addmm(list(node.users.keys()))
                if not result_node:
                    addmm_groups.append([])
                    break
                node = result_node

    # kwargs in the form (name, default_value, idx_in_args)
    kwargs = [("beta", 1.0, -3), ("alpha", 1.0, -2), ("fuse", 0, -1)]

    group_idx = 0
    for group in addmm_groups:
        if len(group) > 1:
            last_addmm = group[-1]
            group_op_args = [[] for _ in range(6)]

            for addmm in group:
                for idx, arg in enumerate(addmm.args):
                    group_op_args[idx].append(arg)

                for kwarg in kwargs:
                    if kwarg[0] in addmm.kwargs:
                        group_op_args[kwarg[2]].append(addmm.kwargs[kwarg[0]])
                    else:
                        group_op_args[kwarg[2]].append(kwarg[1])

            continue_status = False

            for arg in group_op_args[0]:
                # The first set of arguments always denotes the self/bias
                # component of addmm, whenever they get translated from
                # Linear module. These set of arguments must always be
                # placeholders as these do not serve as inputs to the matrix
                # multiplication that takes place in the addmm function.
                # Sequence where the first set of arguments is not a
                # placeholder is not a valid sequence of Linear layers in the
                # model.
                if arg.op != "placeholder":
                    logger.info(
                        "GroupMLP fusion not possible with the current \
                            sequence of addmm layers"
                    )
                    continue_status = True
                    break

            if continue_status:
                continue

            group_op_args[1] = group_op_args[1][0]

            last_addmm.args = tuple(group_op_args)
            # Need to think of a good logic here. This is being done to ensure that
            # kwargs are not present, since they are being taken care by
            # group_op_args
            last_addmm.kwargs = {}
            last_addmm.target = zt_ops.zendnn_vertical_mlp_group.default

            # Name is form group_mlp_{Op Number}
            if group_idx:
                last_addmm.name = f"group_mlp_{group_idx}"
            else:
                last_addmm.name = "group_mlp"

            group_idx += 1

            for addmm in group[::-1]:
                if addmm != last_addmm:
                    fx_graph.graph.erase_node(addmm)

    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()

    return fx_graph
