# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import operator

# import the custom logging module
from ._logging import get_logger
from ._util import save_graph

# make a logger for this file
logger = get_logger(__name__)


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


def replace_addmm_for_1d_bias(node, fx_graph):
    is_fake_tensor = bool(node.args[0].meta)
    # arg node in fx_graph generated through torch.compile will be fake tensor
    if is_fake_tensor and node.args[0].meta["val"].ndim == 1:
        # 1d bias
        return torch.ops.zentorch.zendnn_addmm_1dbias
    # while arg node in fx_graph generated through make_fx will not be fake tensor
    elif not is_fake_tensor and fx_graph._parameters[node.args[0].target].ndim == 1:
        # 1d bias
        return torch.ops.zentorch.zendnn_addmm_1dbias
    else:
        # addmm(2d)
        return torch.ops.zentorch.zendnn_addmm


def replace_with_zendnn_embedding(node, fx_graph):
    # Currently zendnn_embedding op only accepts 1-D inputs
    # which is predominantly evident in RecSys models. The
    # embedding op in Langauge models like Bert work with
    # 2-D inputs. In such cases, we do not replace
    # aten embedding with zendnn embedding. The replacement
    # is taken care by getting the input shapes from the graph.
    is_fake_tensor = bool(node.args[1].meta)
    dims = None
    indices_arg = node.args[1]
    if is_fake_tensor:
        dims = indices_arg.meta["val"].ndim
    else:
        dims = fx_graph._parameters[indices_arg.target].ndim

    if dims == 1:
        return True
    else:
        return False


def replace_with_zendnn_op(fx_graph):
    op_dict = {
        "aten::_embedding_bag": (
            torch.ops.zentorch.zendnn_embedding_bag,
            "zendnn_embedding_bag",
        ),
        "aten::embedding": (torch.ops.zentorch.zendnn_embedding, "zendnn_embedding"),
        "aten::mm": (torch.ops.zentorch.zendnn_mm, "zendnn_mm"),
        "aten::bmm": (torch.ops.zentorch.zendnn_bmm, "zendnn_bmm"),
        "aten::addmm": (torch.ops.zentorch.zendnn_addmm, "zendnn_addmm"),
        "aten::baddbmm": (torch.ops.zentorch.zendnn_baddbmm, "zendnn_baddbmm"),
    }
    # Loop through the nodes in fx_graph.graph
    for node in fx_graph.graph.nodes:
        # Checking for op default implementation to be replaced.
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.name() in op_dict.keys()
        ):
            op_name = node.target.name()
            logger.info(
                "Now replacing default "
                + op_name
                + " with "
                + op_dict[op_name][1]
                + "!"
            )

            if node.target == torch.ops.aten.addmm.default:
                node.target = replace_addmm_for_1d_bias(node, fx_graph)
            else:
                node.target = op_dict[op_name][0]

            if (
                node.target == torch.ops.aten.embedding.default
                and replace_with_zendnn_embedding(node, fx_graph)
            ):
                node.target = op_dict[op_name][0]

    return fx_graph


def set_relu_mm_fusion_kwargs(node):
    return {**node.kwargs, "fuse": 1}


def set_gelu_mm_fusion_kwargs(node):
    fuse = 3
    if bool(node.next.kwargs) and node.next.kwargs["approximate"] == "tanh":
        fuse = 2
    return {**node.kwargs, "fuse": fuse}


# create dict according to fuse
op_eltwise_pattern = {
    "zentorch.zendnn_mm": {
        "aten.relu.default": set_relu_mm_fusion_kwargs,
        "aten.relu_.default": set_relu_mm_fusion_kwargs,
        "aten.gelu.default": set_gelu_mm_fusion_kwargs,
        "aten.gelu_.default": set_gelu_mm_fusion_kwargs,
    },
    "zentorch.zendnn_addmm": {
        "aten.relu.default": set_relu_mm_fusion_kwargs,
        "aten.relu_.default": set_relu_mm_fusion_kwargs,
        "aten.gelu.default": set_gelu_mm_fusion_kwargs,
        "aten.gelu_.default": set_gelu_mm_fusion_kwargs,
    },
    "zentorch.zendnn_addmm_1dbias": {
        "aten.relu.default": set_relu_mm_fusion_kwargs,
        "aten.relu_.default": set_relu_mm_fusion_kwargs,
        "aten.gelu.default": set_gelu_mm_fusion_kwargs,
        "aten.gelu_.default": set_gelu_mm_fusion_kwargs,
    },
}
# for now add is not added as post op that's why I created this pattern

op_add_pattern = [
    ("zentorch.zendnn_bmm", "aten.add.Tensor"),
    ("zentorch.zendnn_mm", "aten.add.Tensor"),
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
        if (str(node.target), str(node.next.target)) in op_add_pattern:
            logger.info(
                "Fusing the "
                + str(node.target)[9:]
                + "->"
                + str(node.next.target.name())[6:]
                + " in fx graph"
            )
            for add_tensor in node.next.args:
                if add_tensor != node:
                    # by *node.args we can append all the arguments
                    new_args = (add_tensor, *node.args)
            node.args = new_args
            node.next.replace_all_uses_with(node)
            fx_graph.graph.erase_node(node.next)

            if node.target == torch.ops.zentorch.zendnn_mm:
                node.target = replace_addmm_for_1d_bias(node, fx_graph)
            else:
                node.target = torch.ops.zentorch.zendnn_baddbmm
        # check the pattern for relu/gelu
        if str(node.target) in op_eltwise_pattern:
            op_dict = op_eltwise_pattern[str(node.target)]
            if str(node.next.target) in op_dict:
                node.kwargs = op_dict[str(node.next.target)](node)
                node.next.replace_all_uses_with(node)
                fx_graph.graph.erase_node(node.next)
    logger.info("Recompiling the fx_graph with fusion changes made.")
    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()
    return fx_graph


def emb_ops_horizontal_fusion(fx_g):
    logger.info("Fusing horizontal parallel ops.")
    groups = {}

    for node in fx_g.graph.nodes:
        if node.target in (
            torch.ops.zentorch.zendnn_embedding_bag,
            torch.ops.zentorch.zendnn_embedding,
        ):
            users = list(node.users.keys())
            user_node = None

            if node.target == torch.ops.zentorch.zendnn_embedding:
                if len(users) == 1:
                    user_node = users[0]
            elif node.target == torch.ops.zentorch.zendnn_embedding_bag:
                for user in users:
                    if user_node is None and len(user.users.keys()) == 1:
                        user_node = user
                    elif user_node is not None and len(user.users.keys()) == 1:
                        user_node = None
                        break

            if user_node is not None:
                common_output_node = None

                if node.target == torch.ops.zentorch.zendnn_embedding:
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
                elif node.target == torch.ops.zentorch.zendnn_embedding_bag:
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
                for i in range(len(node.args)):
                    if node.args[i] is False:
                        list_new_args[i].append(0)
                    elif node.args[i] is True:
                        list_new_args[i].append(1)
                    else:
                        list_new_args[i].append(node.args[i])
                if len(node.args) == 2:
                    list_new_args[2].append(-1)
                    list_new_args[3].append(0)
                    list_new_args[4].append(0)
                if len(node.args) == 3:
                    list_new_args[3].append(0)
                    list_new_args[4].append(0)
                if len(node.args) == 7:
                    list_new_args[7].append(0)
                    list_new_args[8].append(-1)
                if len(node.args) == 8:
                    list_new_args[8].append(-1)

                if node.target == torch.ops.zentorch.zendnn_embedding_bag:
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
                if last_node.target == torch.ops.zentorch.zendnn_embedding:
                    last_node.target = torch.ops.zentorch.zendnn_custom_embedding_group
                elif last_node.target == torch.ops.zentorch.zendnn_embedding_bag:
                    last_node.target = (
                        torch.ops.zentorch.zendnn_custom_embedding_bag_group
                    )
            elif op_count == 1:
                last_node.target = node.target

            if node.target == torch.ops.zentorch.zendnn_custom_embedding_group:
                common_output_node = list(last_node.users.keys())[0]
                if len(common_output_node.args) == 2:
                    common_output_node.args = (last_node, common_output_node.args[1])
                else:
                    common_output_node.args = (last_node,)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    save_graph(fx_g, "zen_groupOp_model")

    return fx_g
