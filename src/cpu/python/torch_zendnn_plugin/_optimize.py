# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
    with zendnn implementation of respective ops
    """
    logger.info("Optimizing the fx_graph with zentorch ops.")

    # Dumping of the native graph in svg format
    save_graph(fx_graph, "native_model")

    # Loop through the nodes in fx_graph.graph
    for node in fx_graph.graph.nodes:
        op_dict = {
            "aten::_embedding_bag": [
                torch.ops.zentorch.zendnn_embedding_bag,
                "zendnn_embedding_bag",
            ],
            "aten::mm": [torch.ops.zentorch.zendnn_mm, "zendnn_mm"],
            "aten::bmm": [torch.ops.zentorch.zendnn_bmm, "zendnn_bmm"],
            "aten::addmm": [torch.ops.zentorch.zendnn_addmm, "zendnn_addmm"],
            "aten::baddbmm": [torch.ops.zentorch.zendnn_baddbmm, "zendnn_baddbmm"],
        }

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
            node.target = op_dict[op_name][0]

    logger.info("Recompiling the fx_graph with op replacement changes made.")
    # Recompile the fx_graph with changes made
    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()

    optimized_graph = op_fusion(fx_graph)

    # Dumping of the optimized graph in svg format
    save_graph(fx_graph, "zen_optimized_model")

    return optimized_graph


def op_fusion(fx_graph):
    """
    op_fusion:
    takes in the fx_graph and fuses some of the native ops
    with zendnn implementation of respective op fusions
    """
    logger.info("Fusing the zentorch ops in fx_graph.")
    # Loop through the nodes in fx_graph.graph
    for node in fx_graph.graph.nodes:
        if (
            node.target == torch.ops.zentorch.zendnn_mm
            or node.target == torch.ops.zentorch.zendnn_addmm
        ):
            if len(node.users) > 1:  # Output of node is used by other nodes
                continue
            # mm->relu or addmm->relu fusion
            if (
                isinstance(node.next.target, torch._ops.OpOverload)
                and node.next.target.name() == "aten::relu"
            ):
                logger.info("Fusing the mm->relu or addmm->relu in fx_graph.")
                new_kwargs = {**node.kwargs, "fuse_relu": True}
                node.kwargs = new_kwargs
                logger.info(
                    "Replacing the next node[relu] with current " "node from the graph."
                )
                node.next.replace_all_uses_with(node)
                logger.info("Removing the next node[relu] from the graph.")
                fx_graph.graph.erase_node(node.next)
                logger.info("Fused the mm->relu or addmm->relu in fx_graph.")
        if (
            node.target == torch.ops.zentorch.zendnn_mm
            or node.target == torch.ops.zentorch.zendnn_bmm
        ):
            if len(node.users) > 1:  # Output of node is used by other nodes
                continue
            # mm->add or bmm->add fusion
            if (
                isinstance(node.next.target, torch._ops.OpOverload)
                and node.next.target.name() == "aten::add.Tensor"
            ):
                logger.info("Fusing the mm->add or bmm->add in fx_graph.")
                for add_tensor in node.next.args:
                    if add_tensor != node:
                        new_args = (add_tensor, *node.args)
                node.args = new_args
                logger.info(
                    "Replacing the next node[add] with current " "node from the graph."
                )
                node.next.replace_all_uses_with(node)
                logger.info("Removing the next node[add] from the graph.")
                fx_graph.graph.erase_node(node.next)
                if node.target == torch.ops.zentorch.zendnn_mm:
                    logger.info("Fused the mm->add to addmm in fx_graph.")
                    node.target = torch.ops.zentorch.zendnn_addmm
                    # [mm->add]->relu fusion
                    if len(node.users) > 1:  # Output of node is used by other nodes
                        continue
                    elif (
                        isinstance(node.next.target, torch._ops.OpOverload)
                        and node.next.target.name() == "aten::relu"
                    ):
                        logger.info("Fusing the [mm->add]->relu fusion in fx_graph.")
                        new_kwargs = {**node.kwargs, "fuse_relu": True}
                        node.kwargs = new_kwargs
                        logger.info(
                            "Replacing the next node[relu] with current "
                            "node from the graph."
                        )
                        node.next.replace_all_uses_with(node)
                        logger.info("Removing the next node[relu] from the graph.")
                        fx_graph.graph.erase_node(node.next)
                        logger.info("Fused the [mm->add]->relu in fx_graph.")
                else:
                    logger.info("Fused the bmm->add to baddbmm in fx_graph.")
                    node.target = torch.ops.zentorch.zendnn_baddbmm

    logger.info("Recompiling the fx_graph with fusion changes made.")
    fx_graph.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_graph.recompile()

    return fx_graph


def replace_emb_bag(fx_g):
    eb_groups = {}

    for node in fx_g.graph.nodes:
        # This function is intended to be used after using the optimize
        # function. So, the zendnn eb bag is being searched for replacement.
        if (isinstance(node.target, torch._ops.OpOverloadPacket)
           and node.target == torch.ops.zentorch.zendnn_embedding_bag):
            users = list(node.users.keys())

            user_node = None
            for user in users:
                if user_node is None and len(user.users.keys()) == 1:
                    user_node = user
                elif user_node is not None and len(user.users.keys()) == 1:
                    user_node = None
                    break

            if user_node is not None:
                common_output_node = list(user_node.users.keys())[0]
                # only if the eb bags have one output and it is common,
                # they can be replaced
                if common_output_node.name in eb_groups:
                    eb_groups[common_output_node.name].append(node)
                else:
                    eb_groups[common_output_node.name] = [node]

    for group in eb_groups:

        embedding_bag_op_count = 0
        list_new_args = [[], [], [], [], [], [], [], [], []]

        for node in eb_groups[group]:
            len_node_args = len(node.args)
            # _embedding_bag function prototype looks as below:
            # _embedding_bag(weight, indices, offsets,
            # scale_grad_by_freq, mode, sparse, per_sample_weights,
            # include_last_offset, padding_idx)
            for i in range(len_node_args):
                if (node.args[i] is False):
                    list_new_args[i].append(0)
                elif (node.args[i] is True):
                    list_new_args[i].append(1)
                else:
                    list_new_args[i].append(node.args[i])
            # by default make_fx passes 7 args for _embedding_bag
            # so we will have to pass deafult values for next 2 args
            if (len_node_args == 7):
                list_new_args[7].append(0)
                list_new_args[8].append(-1)
            if (len_node_args == 8):
                list_new_args[8].append(-1)

            if (embedding_bag_op_count == 0):
                first_emb_node = node
            else:
                # moving all nodes which are args to current _embedding_bag
                # node before the first_emb_node
                for temp_node in node.all_input_nodes[::-1]:
                    first_emb_node.prepend(temp_node)

                # output of _embedding_bag is tuple of 4 tensors but for
                # zendnn_custom_embedding_bag_group output is single TensorList
                # so we will change all indices for all uses of current
                # _embedding_bag node accordingly
                total_prev_outputs_emb_bag = embedding_bag_op_count * 4
                # assuming that tupled output of _embedding_bag is only
                # being used by 4 getitem nodes
                for temp_node in list(node.users.keys()):
                    if (temp_node.target == operator.getitem):
                        temp_node_args = (temp_node.args[0], temp_node.args[1]
                                          + total_prev_outputs_emb_bag)
                        temp_node.args = temp_node_args

                # Replacing the all uses of current _embedding_bag node
                # with first_emb_node
                node.replace_all_uses_with(first_emb_node)
                # Removing the current _embedding_bag node from the graph
                fx_g.graph.erase_node(node)

            embedding_bag_op_count += 1

        if (embedding_bag_op_count > 1):
            first_emb_node.args = tuple(list_new_args)
            target = torch.ops.zentorch.zendnn_custom_embedding_bag_group
            first_emb_node.target = target
        elif (embedding_bag_op_count == 1):
            first_emb_node.target = torch.ops.zentorch.zendnn_embedding_bag

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    # Dumping of the graph with group EB op in svg format
    save_graph(fx_g, "zen_groupEB_op_model")

    return fx_g
