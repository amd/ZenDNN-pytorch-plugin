# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import operator
import zentorch._C # noqa

# import the custom logging module
from ._logging import get_logger
from ._util import save_graph

# make a logger for this file
logger = get_logger(__name__)

at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch


def optimize(fx_graph):
    """
    optimize:
    takes in the fx_graph and replaces some of the native ops
    with zendnn implementation of respective ops and fusion
    few ops
    """
    # Dumping of the native graph in svg format
    save_graph(fx_graph, "native_model")

    logger.info("Optimizing the fx_graph with zentorch ops.")

    # replacing ops to zendnn ops
    fx_graph = replace_with_zendnn_op(fx_graph)

    optimized_graph = zendnn_op_fusion(fx_graph)

    groupEmbedOps_graph = emb_ops_horizontal_fusion(optimized_graph)

    # Dumping of the optimized graph in svg format
    save_graph(groupEmbedOps_graph, "zen_optimized_model")

    return groupEmbedOps_graph


def is_arg_1d_tensor(fx_graph, node, arg_index):
    is_fake_tensor = bool(node.args[arg_index].meta)

    if is_fake_tensor:
        # arg node in fx_graph generated through torch.compile will be fake tensor
        dims = node.args[arg_index].meta["val"].ndim
    else:
        # while arg node in fx_graph generated through make_fx will not be fake tensor
        dims = fx_graph._parameters[node.args[arg_index].target].ndim

    if dims == 1:
        return True
    else:
        return False


# checks the two conditions for arg nodes datatypes
# the tensor to check is either directly accessible through
# parameters of fx_graph or is stored as fake in meta dict.
def is_arg_dtype_bfloat16(fx_graph, node, arg_index):
    is_fake_tensor = bool(node.args[arg_index].meta)

    if is_fake_tensor:
        # arg node in fx_graph generated through torch.compile will be fake tensor
        arg_dtype = node.args[arg_index].meta["val"].dtype
    else:
        # while arg node in fx_graph generated through make_fx will not be fake tensor
        arg_dtype = fx_graph._parameters[node.args[arg_index].target].dtype

    if arg_dtype == torch.bfloat16:
        return True
    else:
        return False


def is_embedding_bag_op_replacable(fx_graph, node):
    if is_arg_dtype_bfloat16(fx_graph, node, 0):
        logger.warning(
            "embedding_bag op will not be replaced as"
            + " zentorch doesn't support bf16 with it yet!"
        )
        # don't replace embedding bag if autocast is enabled or if the model
        # is mixed precision as zendnn doesn't support it w/ bf16
        return False
    else:
        return True


def is_embedding_op_replacable(fx_graph, node):
    if is_arg_dtype_bfloat16(fx_graph, node, 0):
        logger.warning(
            "embedding op will not be replaced as"
            + " zentorch doesn't support bf16 with it yet!"
        )
        # don't replace embedding if autocast is enabled or if the model
        # is mixed precision as zendnn doesn't support it w/ bf16
        return False
    else:
        # Currently zendnn_embedding op only accepts 1-D inputs
        # which is predominantly evident in RecSys models. The
        # embedding op in Langauge models like Bert work with
        # 2-D inputs. In such cases, we do not replace
        # aten embedding with zendnn embedding. The replacement
        # is taken care by getting the input shapes from the graph.
        # returns true if inputs to embedding are 1-D
        return is_arg_1d_tensor(fx_graph, node, 1)


def is_bias_1d_tensor(fx_graph, node):
    # checks if self/bias tensor is 1-d or not
    # returns true if 1d bias tensor
    return is_arg_1d_tensor(fx_graph, node, 0)


def replace_with_zendnn_op(fx_graph):
    op_dict = {
        at_ops._embedding_bag.default: (
            zt_ops.zendnn_embedding_bag.default, is_embedding_bag_op_replacable,
        ),
        at_ops.embedding.default: (
            zt_ops.zendnn_embedding.default, is_embedding_op_replacable,
        ),
        at_ops.mm.default: (zt_ops.zendnn_mm.default, None),
        at_ops.bmm.default: (zt_ops.zendnn_bmm.default, None),
        at_ops.addmm.default: (zt_ops.zendnn_addmm.default, None),
        at_ops.baddbmm.default: (
            zt_ops.zendnn_baddbmm.default, None,
        ),
    }
    # Loop through the nodes in fx_graph.graph
    # Replacing aten ops with respective zendnn ops
    for node in fx_graph.graph.nodes:
        # Checking for op default implementation to be replaced.
        if node.target in op_dict.keys():
            target_op = node.target
            if op_dict[target_op][1] is not None:
                if op_dict[target_op][1](fx_graph, node):
                    logger.info(
                        "Now replacing default "
                        + str(target_op)
                        + " with "
                        + str(op_dict[target_op][0])
                        + "!"
                    )
                    node.target = op_dict[target_op][0]
                else:
                    logger.info(
                        "Not able to replace default "
                        + str(target_op)
                        + " with "
                        + str(op_dict[target_op][0])
                        + " due to non-fulfilment of the condition."
                    )
            else:
                logger.info(
                    "Now replacing default "
                    + str(target_op)
                    + " with "
                    + str(op_dict[target_op][0])
                    + "!"
                )
                node.target = op_dict[target_op][0]

            # currently only zendnn_addmm is the conditional zendnn op
            if node.target == zt_ops.zendnn_addmm.default:
                if is_bias_1d_tensor(fx_graph, node):
                    node.target = zt_ops.zendnn_addmm_1dbias.default

    return fx_graph


def set_relu_mm_fusion_kwargs(node):
    return {**node.kwargs, "fuse": 1}


def set_gelu_mm_fusion_kwargs(node):
    fuse = 3
    if bool(node.next.kwargs) and node.next.kwargs["approximate"] == "tanh":
        fuse = 2
    return {**node.kwargs, "fuse": fuse}


# create dict according to fuse
op_eltwise_pattern = dict.fromkeys(
    (zt_ops.zendnn_mm.default,
     zt_ops.zendnn_addmm.default,
     zt_ops.zendnn_addmm_1dbias.default),
    {at_ops.relu.default: set_relu_mm_fusion_kwargs,
     at_ops.relu_.default: set_relu_mm_fusion_kwargs,
     at_ops.gelu.default: set_gelu_mm_fusion_kwargs,
     at_ops.gelu_.default: set_gelu_mm_fusion_kwargs}
)
# for now add is not added as post op that's why I created this pattern

op_add_pattern = [
    (zt_ops.zendnn_bmm.default, at_ops.add.Tensor),
    (zt_ops.zendnn_mm.default, at_ops.add.Tensor),
]


def zendnn_op_fusion(fx_graph):
    """
    zendnn_op_fusion:
    takes in the fx_graph and fuses some of the native ops
    with zendnn implementation of respective op fusions
    """
    logger.info("Fusing the zentorch ops in fx graph.")
    # Loop through the nodes in fx_graph.graph
    for node in fx_graph.graph.nodes:
        if len(node.users) > 1:  # Output of node is used by other nodes
            continue
        # check the pattern for mm->add or bmm->add
        if (node.target, node.next.target) in op_add_pattern:
            logger.info(
                "Fusing the "
                + str(node.target)
                + "->"
                + str(node.next.target)
                + " in fx graph"
            )
            for add_tensor in node.next.args:
                if add_tensor != node:
                    # by *node.args we can append all the arguments
                    new_args = (add_tensor, *node.args)
            node.args = new_args
            node.next.replace_all_uses_with(node)
            fx_graph.graph.erase_node(node.next)

            if node.target == zt_ops.zendnn_mm.default:
                if is_bias_1d_tensor(fx_graph, node):
                    node.target = zt_ops.zendnn_addmm_1dbias.default
                else:
                    node.target = zt_ops.zendnn_addmm.default
            else:
                node.target = zt_ops.zendnn_baddbmm.default
        # check the pattern for relu/gelu
        if node.target in op_eltwise_pattern:
            eltwise_op_dict = op_eltwise_pattern[node.target]
            if node.next.target in eltwise_op_dict:
                node.kwargs = eltwise_op_dict[node.next.target](node)
                node.next.replace_all_uses_with(node)
                fx_graph.graph.erase_node(node.next)

    logger.info("Recompiling the fx_graph with fusion changes made.")
    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()
    return fx_graph


def emb_ops_horizontal_fusion(fx_g):
    logger.info("Fusing horizontal parallel ops.")
    zentorch_embed_ops_dict = {
        zt_ops.zendnn_embedding_bag.default :
            zt_ops.zendnn_custom_embedding_bag_group.default,
        zt_ops.zendnn_embedding.default :
            zt_ops.zendnn_custom_embedding_group.default,
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
                    if node_args_len == 2:
                        list_new_args[2].append(-1)
                        list_new_args[3].append(0)
                        list_new_args[4].append(0)
                    elif node_args_len == 3:
                        list_new_args[3].append(0)
                        list_new_args[4].append(0)
                    elif node_args_len == 4:
                        list_new_args[4].append(0)
                elif node.target == zt_ops.zendnn_embedding_bag.default:
                    if node_args_len == 3:
                        list_new_args[3].append(0)
                        list_new_args[4].append(0)
                        list_new_args[5].append(0)
                        list_new_args[6].append(None)
                        list_new_args[7].append(0)
                        list_new_args[8].append(-1)
                    elif node_args_len == 4:
                        list_new_args[4].append(0)
                        list_new_args[5].append(0)
                        list_new_args[6].append(None)
                        list_new_args[7].append(0)
                        list_new_args[8].append(-1)
                    elif node_args_len == 5:
                        list_new_args[5].append(0)
                        list_new_args[6].append(None)
                        list_new_args[7].append(0)
                        list_new_args[8].append(-1)
                    elif node_args_len == 6:
                        list_new_args[6].append(None)
                        list_new_args[7].append(0)
                        list_new_args[8].append(-1)
                    elif node_args_len == 7:
                        list_new_args[7].append(0)
                        list_new_args[8].append(-1)
                    elif node_args_len == 8:
                        list_new_args[8].append(-1)

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

            if node.target == zt_ops.zendnn_custom_embedding_group.default:
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

    save_graph(fx_g, "zen_groupOp_model")

    return fx_g
