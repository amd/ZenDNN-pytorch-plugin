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
    logger.info("Fusing horizontal parallel embedding ops.")
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
                            logger.warning(
                                "Cannot fuse embedding bag and embedding with "
                                + "common node. This is because of the function "
                                + "prototype difference between the "
                                + "aten.embedding and aten.embeddingbag ops and "
                                + "their corresponding zentorch group ops!"
                            )
                            return fx_g
                        groups[node_name]["nodes"].append(node)
                    else:
                        groups[node_name] = {
                            "common_output_node_target": common_output_node.target,
                            "type": "embedding",
                            "nodes": [node],
                        }
                elif node.target == zt_ops.zendnn_embedding_bag.default:
                    common_output_node = list(user_node.users.keys())[0]
                    node_name = common_output_node.name
                    if node_name in groups:
                        if groups[node_name]["type"] == "embedding":
                            logger.warning(
                                "Cannot fuse embedding bag and embedding with "
                                + "common node. This is because of the function "
                                + "prototype difference between the "
                                + "aten.embedding and aten.embeddingbag ops and "
                                + "their corresponding zentorch group ops."
                            )
                            return fx_g
                        groups[node_name]["nodes"].append(node)
                    else:
                        groups[node_name] = {
                            "common_output_node_target": common_output_node.target,
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
        #  strict checks for common output node to be concat node
        if (len(groups[group]["nodes"]) > 1) and (
            groups[group]["common_output_node_target"]
        ) == at_ops.cat.default:
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
                nodes_list = []

                idx = 0
                # Since the common output node will always be a cat node, the
                # first argument in the cat node is always a list of tensors
                # that are supposed to be concatenated and the second argument
                # is an integer denoting the dimension across which the
                # concatenation must happen. So, the following for loop which
                # is iterating over the common_output_node.args[0] is valid.
                for arg_node in common_output_node.args[0]:
                    # Since replacement of all occurences of all embedding ops
                    # with the last embedding op takes place and the target is
                    # updated, the condition looks for the target equal to the
                    # zendnn_horizontal_embedding_group, rather than
                    # zendnn_embedding.
                    if (
                        arg_node.target
                        == zt_ops.zendnn_horizontal_embedding_group.default
                    ):
                        new_node = fx_g.graph.create_node(
                            op="call_function",
                            target=operator.getitem,
                            args=(last_node, idx),
                        )
                        last_node.append(new_node)
                        nodes_list.append(new_node)
                        idx += 1
                    else:
                        nodes_list.append(arg_node)

                # Here, the creation of the new args happens and only the first
                # argument is changed (which is the list of input nodes).
                new_args = list(common_output_node.args)
                new_args[0] = nodes_list
                common_output_node.args = tuple(new_args)

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

    logger.info("Fusing vertical contiguous addmm ops.")

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
                    logger.warning(
                        "GroupMLP fusion not possible with the current "
                        + "sequence of addmm layers"
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


def horizontal_mlp_fusion(fx_g):
    # Fusing parallel matmuls in attention block which constitute
    # key query value pair for LLM's.
    logger.info("Fusing horizontal Matmul ops.")
    groups = {}
    user_node_list = []
    for node in fx_g.graph.nodes:
        node_name = node.name
        # Pattern check begins with finding the common node
        # for the horizontal matmul pattern
        if len(node.users) >= 3:
            for user in node.users.keys():
                # Skipping if there are no users present or if the user is a return node
                if not bool(user.users.keys()):
                    continue
                user_node = list(user.users.keys())[0]
                # Skipping fusion if args of view are not placeholders or constants.
                #
                view_arg_check = False
                if user.target == torch.ops.aten.view.default:
                    for node_arg in user.args[1]:
                        if (
                            type(node_arg) is torch.fx.node.Node
                            and node_arg.op != "placeholder"
                        ):
                            view_arg_check = True
                            logger.warning(
                                "GroupMLP fusion not possible with the current "
                                + "arg list of view node as args of view are not "
                                + "constants or placeholders."
                            )
                if view_arg_check:
                    continue
                # Append addmm/mm nodes to group
                # Check if addmm/mm is unique to the dictionary
                if (
                    user_node.target == torch.ops.zentorch.zendnn_addmm_1dbias.default
                    and user_node not in user_node_list
                ):
                    groups.setdefault(node_name, {"nodes": []})["nodes"].append(
                        user_node
                    )
                    groups[node_name].update(
                        {"beta": (1.0, -4), "alpha": (1.0, -3), "fuse": (0, -2)}
                    )
                    user_node_list.append(user_node)
                elif (
                    user_node.target == torch.ops.zentorch.zendnn_mm.default
                    and user_node not in user_node_list
                ):
                    groups.setdefault(node_name, {"nodes": []})["nodes"].append(
                        user_node
                    )
                    groups[node_name].update(
                        {"beta": (0.0, -4), "alpha": (1.0, -3), "fuse": (0, -2)}
                    )
                    user_node_list.append(user_node)

    # Perform fusion and optimization
    for group in groups.values():
        # Check for attention block matmuls
        # Validate if all K,Q,V are of same target value
        target_values = [(group["nodes"][i]).target for i in range(len(group["nodes"]))]
        # Checking only for K,Q,V pair hence hardcoded to 3
        if len(group["nodes"]) == 3 and len(set(target_values)) == 1:
            group_op_args = [[] for _ in range(7)]
            first_node = group["nodes"][0]
            # Check if node is zendnn_addmm or zendnn_mm.
            # zendnn_mm has only 2 arguments, whereas zendnn_addmm has 3.
            # create an empty node to replicate the arg structure of addmm.
            if len(first_node.args) == 2:
                # take arbitrary shape value for the empty_node being created
                # from the previous node. HF LLMs have view as the previous node.
                empty_shape = [list(first_node.args[0].args[1])]
                empty_node_args = tuple(empty_shape)
                empty_node = fx_g.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.empty.memory_format,
                    args=(empty_node_args),
                )
                first_node.prepend(empty_node)
            # Prepare argument list for the fused op
            for node in group["nodes"]:
                if len(node.args) == 2:
                    group_op_args[0].append(empty_node)
                for i, arg in enumerate(node.args):
                    if len(node.args) == 2:
                        group_op_args[i + 1].append(arg)
                    else:
                        group_op_args[i].append(arg)
                    if arg.target in (
                        torch.ops.aten.view.default,
                        torch.ops.aten.permute.default,
                    ):
                        first_node.prepend(arg)
                # iterate through kwargs to append node's value, if present in graph,
                # else append the default value
                for kwarg in ["fuse", "alpha", "beta"]:
                    if group[kwarg] in node.kwargs:
                        group_op_args[group[kwarg][1]].append(node.kwargs[kwarg])
                    else:
                        group_op_args[group[kwarg][1]].append(group[kwarg][0])
                # Adding flag to recognise zendnn_mm or zendnn_addmm to make
                # appropriate changes in op definition.
                if first_node.target == torch.ops.zentorch.zendnn_addmm_1dbias.default:
                    group_op_args[6].append(0)
                elif first_node.target == torch.ops.zentorch.zendnn_mm.default:
                    group_op_args[6].append(1)
            # Create zendnn_attn_horizontal_mlp_group node
            group_node = fx_g.graph.create_node(
                op="call_function",
                target=torch.ops.zentorch.zendnn_attn_horizontal_mlp_group.default,
                args=tuple(group_op_args),
            )
            first_node.prepend(group_node)

            # Creating getitem nodes to parse the output vector.
            getitem_nodes = []
            for getitem_num in range(3):
                new_node = fx_g.graph.create_node(
                    op="call_function",
                    target=operator.getitem,
                    args=(group_node, getitem_num),
                )
                group_node.append(new_node)  # FX API
                getitem_nodes.append(new_node)  # LIST API

            for i, node in enumerate(group["nodes"]):
                node.replace_all_uses_with(getitem_nodes[i])
                fx_g.graph.erase_node(node)
        else:
            logger.info(
                "Horizontal Group fusion not possible with the current \
                            combination of addmm/mm layers"
            )
    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()
    return fx_g


def eb_group_mlp_group_fusion(fx_graph):

    logger.info(
        "Fusing the horizontally fused EmbeddingBag op and the"
        + " vertically fused MLP op"
    )

    # This function makes changes on the graph after the Horizontal
    # EmbeddingBag Fusion and Vertical MLP fusion is introduced in the graph.
    # This fusion merges the horizontally fused EmbeddingBag op and the
    # vertically fused MLP op into one single op to achieve better compute
    # distribution.

    # The above mentioned fusion happens only when the horizontally fused
    # EmbeddingBag op and the vertically fused MLP op have a common concat node
    concat_nodes = []
    for node in fx_graph.graph.nodes:
        if node.target == torch.ops.aten.cat.default:
            concat_nodes.append(node)

    # Return the index of the required_node in the fx_graph. Since the fx_graph
    # is a doubly linked list, the index of the required_node in a graph makes
    # sense.
    def get_node_index_in_graph(required_node, fx_graph):
        for idx, node in enumerate(fx_graph.graph.nodes):
            if node == required_node:
                return idx

    # EmbeddingBag has four outputs, each of which are get_item nodes in the
    # graph. Group EmbedddingBag considers these get_items from various
    # EmbeddingBags with common output of these get_items being the concate
    # node. So, any get_item nodes that have output as concate node and input
    # as EmbeddingBag, will be merged into one Group EmbedddingBag. So, one
    # concate node cannot have more than one Group EmbedddingBag as the input.

    group_idx = 0
    for node in concat_nodes:
        group_mlp_op, group_eb_op = None, None
        node_input_loop_break = False
        for node_input in node.all_input_nodes:
            # Checking if the one of the inputs to the concate node is
            # vertically fused MLP op. If there are multiple vertically fused
            # MLP ops that are inputs to the interaction node, the last one
            # will be fused with the horizontally fused EmbeddingBag op
            if node_input.target == zt_ops.zendnn_vertical_mlp_group.default:
                group_mlp_op = node_input

            # Here we strictly checking that every get_item node has the
            # horizontally fused EmbeddingBag op as the input. If that is not
            # the case, the fusion does not take place and the control goes to
            # the next interaction node.
            elif node_input.target == operator.getitem:
                for getitem_input in node_input.all_input_nodes:
                    condition = (
                        getitem_input.target
                        == zt_ops.zendnn_horizontal_embedding_bag_group.default
                    )
                    if condition:
                        group_eb_op = getitem_input
                    else:
                        node_input_loop_break = True
                        break

            # Here we have a very strict check, that is concate node under
            # consideration must mandatorily have only horizontally fused
            # EmbeddingBag ops and the vertically fused MLP ops as inputs
            # else we proceed to the next interaction node
            else:
                break

            # If even one of the get_item nodes have a different input than
            # horizontally fused EmbeddingBag op, we proceed to the next
            # interaction node
            if node_input_loop_break:
                logger.warning(
                    "Fusion of horizontally fused EmbeddingBag op and"
                    + " the vertically fused MLP op into one single op is"
                    + f" not possible at the current concate node: {node}!"
                )
                group_mlp_op, group_eb_op = None, None
                break

        if group_eb_op and group_mlp_op:
            fused_op_args = []

            group_mlp_op_idx = get_node_index_in_graph(group_mlp_op, fx_graph)
            group_eb_op_idx = get_node_index_in_graph(group_eb_op, fx_graph)

            start_node = group_mlp_op
            end_node = group_eb_op
            curr_node = start_node.next

            # The update of the target always happens to the vertically fused
            # MLP op. So, whenever horizontally fused EmbeddingBag op comes
            # before vertically fused MLP op, we need to prepend the arguments
            # of horizontally fused EmbeddingBag op with respect to the
            # before vertically fused MLP op
            if group_mlp_op_idx < group_eb_op_idx:
                while curr_node != end_node:
                    group_mlp_op.prepend(curr_node)
                    curr_node = curr_node.next

            for arg in group_eb_op.args:
                fused_op_args.append(arg)
            for arg in group_mlp_op.args:
                fused_op_args.append(arg)
            group_mlp_op.args = tuple(fused_op_args)
            new_node = None

            group_mlp_op.target = torch.ops.zentorch.zendnn_fused_eb_mlp.default

            if group_idx:
                group_mlp_op.name = f"fused_eb_mlp_{group_idx}"
            else:
                group_mlp_op.name = "fused_eb_mlp"
            group_idx += 1

            # Replacing the output of original vertical Group MLP op to a
            # new get_item node. To avoid nested replacement of Group MLP op
            # inside the arguments of the get_item, we are assigning the
            # arguments after the replacement of the Group MLP op with the
            # new node.
            new_node = fx_graph.graph.create_node(
                op="call_function",
                target=operator.getitem,
                args=(),
            )
            group_mlp_op.append(new_node)
            group_mlp_op.replace_all_uses_with(new_node)
            new_node.args = (group_mlp_op, (len(fused_op_args[0]) * 4))

            for user in list(group_eb_op.users.keys()):
                group_mlp_op.append(user)

            # Since we are fusing the horizontally fused EmbeddingBag op with
            # vertically fused MLP op, and the target of the latter is always
            # changed, we replace all the former's uses with the latter and
            # erase the former.
            group_eb_op.replace_all_uses_with(group_mlp_op)
            fx_graph.graph.erase_node(group_eb_op)

    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()

    return fx_graph
