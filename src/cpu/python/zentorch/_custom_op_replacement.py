# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from torch._inductor.pattern_matcher import stable_topological_sort
import os
import operator
from ._utils import counters, find_path

# import the custom logging module
from ._logging import get_logger

# make a logger for this file
logger = get_logger(__name__)
at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch


def has_out_variant_for_all_args(node):
    schema = node.target._schema
    arg_indices = {arg.name: i for i, arg in enumerate(schema.arguments)}
    tensors_idx = arg_indices["tensors"]
    # Can not apply fusion for torch.cat([linear_0, linear_0])
    nodes_visited = []
    for arg in node.args[tensors_idx]:
        # TODO: Handle get item nodes
        if (
            arg.op == "call_function"
            and isinstance(arg.target, torch._ops.OpOverload)
            and hasattr(zt_ops, arg.target._opname)
        ):
            if arg in nodes_visited:
                return False
            op = getattr(zt_ops, arg.target._opname)
            if hasattr(op, "out"):
                logger.info("Found out variant for %s", arg.target._opname)
                nodes_visited.append(arg)
                continue
            else:
                logger.info("No out variant found for %s", arg.target._opname)
                return False
        else:
            return False

    return True


def is_cat_dim_valid_for_folding(node):
    """
    Check if the dimension for cat is valid for fusion.
    For now, there is only support for concat along last dimension.
    """
    input_tensors = node.args[0]
    tensor_dimensions = input_tensors[0].meta["val"].ndim
    # If dim is specified as 0, it comes as a keyword argument
    # else it is the second argument in the args tuple
    dim = node.args[1] if len(node.args) > 1 else 0
    return dim == (tensor_dimensions - 1)


def inplace_cat_fusion(fx_g):
    nodes_to_remove = []
    for node in fx_g.graph.nodes:
        if node.target == torch.ops.aten.cat.default and has_out_variant_for_all_args(
            node
        ):
            if not is_cat_dim_valid_for_folding(node):
                logger.info("Cat node %s dim is not valid for fusion", node)
                continue

            # create the output node
            shape = node.meta["tensor_meta"].shape
            with fx_g.graph.inserting_after(node):
                get_out_node = fx_g.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.empty.memory_format,
                    args=(shape,),
                    kwargs={
                        "dtype": node.meta["val"].dtype,
                        "device": node.meta["val"].device,
                    },
                )

            offset = 0
            for arg in node.args[0]:
                with fx_g.graph.inserting_after(get_out_node):
                    as_strided_node = fx_g.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.as_strided,
                        args=(
                            get_out_node,
                            arg.meta["val"].shape,
                            node.meta["val"].stride(),
                            offset,
                        ),
                        kwargs={},
                    )
                # Offset calculation as per concat along last dimension
                # Hence we do the "is_cat_dim_valid_for_folding" check above
                offset += arg.meta["val"].shape[-1]
                op = getattr(zt_ops, arg.target._opname)
                counters["zentorch"]["out_variant"] += 1
                with fx_g.graph.inserting_after(as_strided_node):
                    new_args = (
                        as_strided_node,
                    ) + arg.args  # Correctly construct the new_args tuple
                    out_node = fx_g.graph.create_node(
                        op="call_function",
                        target=op.out,
                        args=new_args,
                        kwargs=arg.kwargs,
                    )
                arg.replace_all_uses_with(out_node)
                nodes_to_remove.append(arg)

            node.replace_all_uses_with(get_out_node)
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        fx_g.graph.erase_node(node)

    fx_g.graph.lint()
    fx_g.recompile()
    return fx_g


def emb_ops_horizontal_fusion(fx_g):
    logger.info("Fusing horizontal parallel embedding ops.")
    zentorch_embed_ops_dict = {
        zt_ops.zentorch_embedding_bag.default: zt_ops.zentorch_horizontal_embedding_bag_group.default,
        zt_ops.zentorch_embedding.default: zt_ops.zentorch_horizontal_embedding_group.default,
        zt_ops.zentorch_quant_embedding_bag.default: zt_ops.zentorch_horizontal_quant_embedding_bag_group.default,
    }
    groups = {}
    for node in fx_g.graph.nodes:
        if node.target in zentorch_embed_ops_dict:
            users = list(node.users.keys())
            user_node = None
            if len(users) == 1:
                user_node = users[0]
            # TODO
            # we can check here itself if the user node is concat else return
            # the fx graph as is or continue looking for more eb nodes in the
            # group
            if user_node is not None:
                if node.target == zt_ops.zentorch_embedding.default:
                    common_output_node = user_node
                    node_name = common_output_node.name
                    if node_name in groups:
                        if groups[node_name]["type"] in {
                            "embedding_bag",
                            "quant_embedding_bag",
                        }:
                            other_op = groups[node_name]["type"]
                            logger.info(
                                "Cannot fuse %s and embedding with "
                                "common node. This is because of the function "
                                "prototype difference between the "
                                "embedding and %s ops and "
                                "their corresponding zentorch group ops!",
                                other_op,
                                other_op,
                            )
                            return fx_g
                        groups[node_name]["nodes"].append(node)
                    else:
                        groups[node_name] = {
                            "common_output_node": (
                                common_output_node.name,
                                common_output_node.target,
                            ),
                            "type": "embedding",
                            "nodes": [node],
                        }
                # TODO
                # embedding bag and quant embedding bag conditions will mostly
                # have similar code except the group op that we replace with.
                # We can make templatize based on common node.This way pass
                # will be smaller and easier to maintain
                elif node.target == zt_ops.zentorch_embedding_bag.default:
                    common_output_node = user_node
                    node_name = common_output_node.name
                    if node_name in groups:
                        if groups[node_name]["type"] in {
                            "embedding",
                            "quant_embedding_bag",
                        }:
                            other_op = groups[node_name]["type"]
                            logger.info(
                                "Cannot fuse %s and embedding_bag with "
                                "common node. This is because of the function "
                                "prototype difference between the "
                                "embeddingbag and %s ops and "
                                "their corresponding zentorch group ops.",
                                other_op,
                                other_op,
                            )
                            return fx_g
                        groups[node_name]["nodes"].append(node)
                    else:
                        groups[node_name] = {
                            "common_output_node": (
                                common_output_node.name,
                                common_output_node.target,
                            ),
                            "type": "embedding_bag",
                            "nodes": [node],
                        }
                elif node.target == zt_ops.zentorch_quant_embedding_bag.default:
                    common_output_node = user_node
                    node_name = common_output_node.name
                    if node_name in groups:
                        if groups[node_name]["type"] in {"embedding", "embedding_bag"}:
                            other_op = groups[node_name]["type"]
                            logger.info(
                                "Cannot fuse %s and "
                                "quant_embedding_bag with common node. This "
                                "is because of the function prototype "
                                "difference between the quantized embedding "
                                "bag and %s ops and their "
                                "corresponding zentorch group ops.",
                                other_op,
                                other_op,
                            )
                            return fx_g
                        groups[node_name]["nodes"].append(node)
                    else:
                        # TODO: Add a unittest for all scenarios
                        groups[node_name] = {
                            "common_output_node": (
                                common_output_node.name,
                                common_output_node.target,
                            ),
                            "type": "quant_embedding_bag",
                            "nodes": [node],
                        }

    def populate_default_args(list_new_args, type_of_node):
        # Depending on the already existing values of arguments in
        # list_new_args, the remaining arguments are populated for embedding
        # ops. The type_of_node decides whether the arguments that are
        # populated are of embedding op, embeddingbag op or quantized
        # embeddingbag op. Embedding requires two out of 7 arguments
        # mandatorily and other arguments can have either default or user
        # specified values. Similarly, EmbeddingBag requires three out of nine
        # arguments mandatorily and other arguments can either have default or
        # user specified values. Likewise, Quantized EmbeddingBag requires five
        # out of eleven arguments mandatorily and other arguments can either
        # have default of user specified values. The list_new_args list is
        # populated as specified above and returned.
        num_ops = len(list_new_args[0])
        default_args = None
        if type_of_node == "embedding":
            # embedding(
            # weight: Tensor,
            # indices: Tensor,
            # padding_idx: int = -1,
            # scale_grad_by_freq: bool = False,
            # sparse: bool = False,
            # include_last_offset: bool = False
            # )
            default_args = [None, None, -1, 0, 0]
        elif type_of_node == "embedding_bag":
            # embedding_bag(
            # weight: Tensor,
            # indices: Tensor,
            # offsets: Tensor,
            # scale_grad_by_freq: bool = False,
            # mode: int = 0,
            # sparse: bool = False,
            # per_sample_weights: Optional[Tensor] = None,
            # include_last_offset: bool = False
            # )
            default_args = [None, None, None, 0, 0, 0, None, 0, -1]
        elif type_of_node == "quant_embedding_bag":
            # quant_embedding_bag(
            # weight: Tensor,
            # indices: Tensor,
            # offsets: Tensor,
            # num_bits_per_weight: int,
            # output_dtype: ScalarType,
            # scale_grad_by_freq: bool = False,
            # mode: int = 0,
            # sparse: bool = False,
            # per_sample_weights: Optional[Tensor] = None,
            # include_last_offset: bool = False,
            # padding_idx: int = -1
            # )

            # Currently, we only support embedding bag quantized to uint4
            # the op implementation used the num_bits_per_weight in calculation
            # which in this case is 4 (Because we quantized to uint4 and then
            # packed the weights into uint32)
            default_args = [None, None, None, 4, torch.float32, 0, 0, 0, None, 0, -1]
        non_empty_args_idx = 0
        for idx, l in enumerate(list_new_args):
            if len(l) == 0:
                non_empty_args_idx = idx
                break
        for idx in range(non_empty_args_idx, len(default_args)):
            list_new_args[idx] = [default_args[idx] for _ in range(num_ops)]
        return list_new_args

    for group in groups:
        # For FP32 or BF16, the condition for length will be always greater than 1.
        condition_for_length = len(groups[group]["nodes"]) > 1

        # Since there is no support for quant eb, we replace quant_eb with
        # quant_eb_group with singular lists. So, condition for length will be
        # true irrespective of number of quantized embedding ops.
        # TODO
        # A more cleaner approach for this would be to move the implementation
        # of singular quant embedding bag inside the CPP files (EmbedBag.cpp),
        # and call the group quant embeddingbag function internally. Currently,
        # we are replacing even singular quant embeddingbag with group counterpart
        # to avoid the usage of the IntArrayRef and TensorList APIs on the CPP part
        # as there are some unknows with it.
        if groups[group]["type"] == "quant_embedding_bag":
            condition_for_length = True

        # From all the embedding operator variants supported by zentorch, there
        # is only output from it. So, there are no getitem nodes to to direct
        # the output to proper places. So when the group counterpart of the
        # embedding operator is introduced in the graph, it returns a list of
        # tensors. So, the presence of getitem nodes is necessary for output
        # re-direction.
        # TODO
        # Add get-item nodes for all the embedding operator variants and its
        # group counterpart. This will ease our graph passes and make this
        # graph pass more generic.

        #  strict checks for common output node to be concat node
        if condition_for_length and (
            groups[group]["common_output_node"][1] == at_ops.cat.default
        ):
            op_count = 0
            # quant_embedding_bag has more parameters than embedding and
            # embedding_bag. So creating new args list with max of the number
            # of parameters of both ops, which is 11. Empty lists are further
            # removed.
            list_new_args = [[] for _ in range(11)]
            last_node = groups[group]["nodes"][-1]
            for node in groups[group]["nodes"]:
                node_args_len = len(node.args)
                for i in range(node_args_len):
                    if node.args[i] is False:
                        list_new_args[i].append(0)
                    elif node.args[i] is True:
                        list_new_args[i].append(1)
                    else:
                        list_new_args[i].append(node.args[i])
            if node.target == zt_ops.zentorch_embedding.default:
                list_new_args = populate_default_args(
                    list_new_args, type_of_node="embedding"
                )
            elif node.target == zt_ops.zentorch_embedding_bag.default:
                list_new_args = populate_default_args(
                    list_new_args, type_of_node="embedding_bag"
                )
            elif node.target == zt_ops.zentorch_quant_embedding_bag.default:
                list_new_args = populate_default_args(
                    list_new_args, type_of_node="quant_embedding_bag"
                )

                # But before we actually make the list into a singular value,
                # we want to confirm that all the values in the list are equal

                # We want to check the num_bits_per_weight and output dtype for
                # all the quant embedding bag nodes. num_bits_per_weight is at
                # the third position and the output dtype is at the fourth
                # position. Hence the hard-coding.
                continue_status_for_graph_iteration = False
                for idx in [3, 4]:
                    ref_value = list_new_args[idx][0]
                    if not all(i == ref_value for i in list_new_args[idx]):
                        logger.info(
                            "Cannot fuse zentorch quant embedding bags into a "
                            "zentorch_horizontal_quant_embedding_bag_group "
                            "as the non-tensor arguments are different "
                            "across embedding bags."
                        )
                        # Since there can be multiple such sites in the graph where
                        # a group embedding bag can be followed by a cat node, we
                        # dont want to just exit out when a check fails. Instead we
                        # would want to look for next site. So breaking out of the
                        # current for loop and for continuing the next one.
                        continue_status_for_graph_iteration = True
                        break
                if continue_status_for_graph_iteration:
                    continue

                # The group quant embedding bag expects the number of bits per
                # weight as a single element rather the list of elements unlike
                # other arguments. The number of bits per weight is at the 3rd
                # index in the argument list. So, the hard-coding of 3 in the
                # subsequent lines.
                list_new_args[3] = list_new_args[3][0]
                # The group quant embedding bag expects the output dtype as a
                # single element rather the list of elements unlike other
                # arguments. So, the output dtype is at the 4th index in the
                # argument list. So, the hard-coding of 4 in the subsequent lines.
                list_new_args[4] = list_new_args[4][0]
            for node in groups[group]["nodes"]:
                if node != last_node:
                    node.replace_all_uses_with(last_node)
                    fx_g.graph.erase_node(node)
                op_count += 1
            if op_count > 1:
                idx = -1
                while len(list_new_args[idx]) == 0:
                    list_new_args.pop()
                last_node.args = tuple(list_new_args)
                if last_node.target in zentorch_embed_ops_dict:
                    last_node.target = zentorch_embed_ops_dict[last_node.target]
            elif op_count == 1:
                last_node.target = node.target

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
                # Since replacement of all occurences of all embedding operator
                # variants with the last embedding operator variant takes place
                # and the target is updated, the condition looks if the target
                # is present in the set of group operator targets, rather than
                # singular operator targets.
                if arg_node.target in zentorch_embed_ops_dict.values():
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

    fx_g.graph.lint()
    fx_g.recompile()

    return fx_g


def group_eb_concat_fusion(fx_graph):
    for node in fx_graph.graph.nodes:
        # Currently the support is only added for horizontal quant embedding
        # bag group
        # TODO
        # Add support for FP32/BF16 horizontal embedding bag group as well
        if node.target == (
            zt_ops.zentorch_horizontal_quant_embedding_bag_group.default
        ):
            # The indices of non-tensor arguments are collected here
            # We want to fuse the cat node with the group embedding bag only
            # when the non tensor arguments of all embedding bags are the same.
            # This is a stricter check just to make sure that the output shapes
            # of all embedding bags are compatible for concatenation, since we
            # will be doing concatenation in the CPP side. We wouldn't be able
            # to add too much flexibility on the CPP side as we can on the
            # Python side. So the stricter check.
            non_tensor_list_arguments_idx = [5, 6, 7, 8, 9, 10]
            continue_status_for_graph_iteration = False
            for idx in non_tensor_list_arguments_idx:
                argument_list = node.args[idx]
                ref_value = argument_list[0]
                # Making sure that the non-tensor arguments for all the
                # embeddingbags are the same.
                if not all(i == ref_value for i in argument_list):
                    logger.info(
                        "Cannot fuse "
                        "zentorch_horizontal_quant_embedding_bag_group and "
                        "concat node as the non-tensor arguments are "
                        "different across embedding bags."
                    )
                    # Since there can be multiple such sites in the graph where
                    # a group embedding bag can be followed by a cat node, we
                    # dont want to just exit out when a check fails. Instead we
                    # would want to look for next site. So breaking out of the
                    # current for loop and for continuing the next one.
                    continue_status_for_graph_iteration = True
                    break
            if continue_status_for_graph_iteration:
                continue

            user_of_getitems_is_cat = False
            cat_node = None
            for getitem_node in node.users.keys():
                users_of_getitem = list(getitem_node.users.keys())
                # The group embedding bag always returns a tuple of outputs.
                # So, the graph always has getitem nodes to transfer the output
                # to correct node in the subsequent graph. So now, here, we
                # want to make sure that the getitem nodes are always a
                # uni-output node. That is the output of the getitem node is
                # always used by only one node, which is cat node in this case.
                # This is because, when we fuse the group embedding bag with
                # the cat node, we will be removing the getitems and if any of
                # the getitems have mode than one user, then linking that
                # output in a concatenated tensor will be really difficult. So
                # this strict check is added just to make sure that all the
                # getitem nodes are uni-output.
                if len(users_of_getitem) == 1:
                    user_of_getitem = users_of_getitem[0]
                    # The CPP part provides the fusion for only a cat node
                    # fusion as this is the pattern found the DLRM models and
                    # also cat operation is simpler to perform.
                    # TODO
                    # See if this fusion can be extended to some more
                    # operations like torch.sum, torch.stack etc.
                    if user_of_getitem.target == at_ops.cat.default:
                        user_of_getitems_is_cat = True
                        cat_node = user_of_getitem
                else:
                    logger.info(
                        "Cannot fuse "
                        "zentorch_horizontal_quant_embedding_bag_group and "
                        "concat node as the output of one/more embedding "
                        "bags is being used by more than one node."
                    )
                    # Since there can be multiple such sites in the graph where
                    # a group embedding bag can be followed by a cat node, we
                    # dont want to just exit out when a check fails. Instead we
                    # would want to look for next site. So breaking out of the
                    # current for loop and for continuing the next one.
                    continue_status_for_graph_iteration = True
                    break
            if continue_status_for_graph_iteration:
                continue

            if user_of_getitems_is_cat:
                cat_dim = 0
                cat_node_args = cat_node.args
                # Cat node has maximum of two arguments, the first is the
                # tensor list and the second one is the dimension along which
                # the concatenation must happen. So, the assumption of the cat
                # dim is equal to 0 is made if the arguments length is equal to
                # 1, else we extract the cat dim from the second argument.
                if len(cat_node_args) == 2:
                    cat_dim = cat_node_args[1]
                (other_arguments_position, other_arguments, getitem_nodes) = (
                    [],
                    [],
                    [],
                )
                for idx, input_arg in enumerate(cat_node_args[0]):
                    # The basic functionality of this pass is to fuse the group
                    # embedding bag with the subsequent cat node. So, we only
                    # want to remove the getitems (as previously mentioned, the
                    # embeddingbag outputs are received via getitems) that come
                    # from the group embedding bag only. Any other getitem
                    # nodes or other nodes are considered as other arguments to
                    # the the cat node and are kept as it is. The getitems from
                    # the group embedding bag are deleted however, as they no
                    # longer serve any purpose.
                    if (input_arg.target == operator.getitem) and (
                        input_arg.args[0] == node
                    ):
                        getitem_nodes.append(input_arg)
                    else:
                        other_arguments.append(input_arg)
                        other_arguments_position.append(idx)

                # As of now, we are assuming that the fx pass will ensure only
                # one extra tensor will be passed in the other_arguments
                # TensorList and that the tensor will be always be concatenated
                # at either zeroth or last position.
                # TODO
                # To futureproof this code and accept as many other arguments
                # and at arbitrary positions in the concatenated tensor, an
                # optimal approach must be found out and designed.

                if len(other_arguments) > 1:
                    continue
                if other_arguments_position[0] == len(getitem_nodes):
                    other_arguments_position[0] = -1

                if other_arguments_position[0] not in {0, -1}:
                    continue

                cat_node.replace_all_uses_with(node)
                fx_graph.graph.erase_node(cat_node)
                for getitem_node in getitem_nodes:
                    fx_graph.graph.erase_node(getitem_node)

                new_args = node.args + (
                    cat_dim,
                    other_arguments_position,
                    other_arguments,
                )
                node.args = new_args
                use_zendnn_eb = os.environ.get("USE_ZENDNN_EB", default="1").lower()
                if use_zendnn_eb in {"0", "false", "f", "no", "n"}:
                    node.target = (
                        zt_ops.zentorch_quant_group_eb_mlp_concat_fbgemm.default
                    )
                else:
                    node.target = (
                        zt_ops.zentorch_quant_group_eb_mlp_concat_zendnn.default
                    )
            else:
                logger.info(
                    "Cannot fuse zentorch_horizontal_quant_embedding_bag_group "
                    "and concat node as the output of one/more embedding "
                    "bags is not being used by concat node."
                )
                continue

    fx_graph.graph.lint()
    fx_graph.recompile()

    return fx_graph


# TODO : Address the case of mutiple child nodes of a qlinear_* op consuming the
# output as the main input (non-postop input).
def qlinear_reorder_optimizations(fx_graph):
    reorder_qlinear_candidates = {
        zt_ops.zentorch_qlinear.default,
        zt_ops.zentorch_qlinear_relu.default,
        zt_ops.zentorch_qlinear_sigmoid.default,
        zt_ops.zentorch_qlinear_mul_add.default,
    }

    def next_user_node(users):
        if len(users) == 1 and users[0].target in reorder_qlinear_candidates:
            return users[0]
        return None

    logger.info("Reorder optimization for serialized qlinear_* ops.")

    # Group a serialized pattern of qlinear_* ops and optimize the
    # dequant-quant operation to a requant operation.
    # TODO : Validate if dictionary with key : node and value : next_node, is
    # a better solution for this optimization.
    qlinear_groups = [[]]
    nodes_traversed = set()
    for node in fx_graph.graph.nodes:
        if node.target in reorder_qlinear_candidates:
            while node not in nodes_traversed:
                qlinear_groups[-1].append(node)
                nodes_traversed.add(node)
                user_node = next_user_node(list(node.users.keys()))
                if not user_node:
                    qlinear_groups.append([])
                    break
                node = user_node

    # Modify the output_dtype and add quant information in predecessor qlinear_*
    # node based on the successor.
    for group in qlinear_groups:
        if len(group) > 1:
            pred_node = group[0]
            for curr_node in group[1:]:
                curr_args = curr_node.args
                # TODO: modify the output dtype comparison after
                # Quark v1.0.0 release.
                # if dtype is not None or
                # if dtype is None then o/p-scales & o/p-zp should also be None
                # and copy all the arguments, then below holds good
                if not bool(pred_node.kwargs) or (
                    "output_dtype" in pred_node.kwargs
                    and pred_node.kwargs["output_dtype"]
                    in (torch.float, torch.bfloat16)
                ):
                    schema = curr_node.target._schema
                    # Map argument names to index
                    arg_indices = {
                        arg.name: i for i, arg in enumerate(schema.arguments)
                    }
                    new_kwargs = dict(pred_node.kwargs)
                    input_scales_idx = arg_indices["input_scales"]
                    input_zero_points_idx = arg_indices["input_zero_points"]
                    new_kwargs["output_dtype"] = (
                        torch.int8
                        if curr_args[input_zero_points_idx] is None
                        else torch.uint8
                    )
                    new_kwargs["output_scales"] = curr_args[input_scales_idx]
                    new_kwargs["output_zero_points"] = curr_args[input_zero_points_idx]
                    pred_node.kwargs = new_kwargs
                    counters["zentorch"]["optimized_reorder"] += 1
                pred_node = curr_node

    fx_graph.graph.lint()
    fx_graph.recompile()
    return fx_graph


def get_fuse_val(target):
    if target in (
        zt_ops.zentorch_mm_relu.default,
        zt_ops.zentorch_addmm_relu.default,
        zt_ops.zentorch_addmm_1dbias_relu.default,
    ):
        return 1
    elif target in (
        zt_ops.zentorch_mm_gelu_tanh.default,
        zt_ops.zentorch_addmm_gelu_tanh.default,
        zt_ops.zentorch_addmm_1dbias_gelu_tanh.default,
    ):
        return 2
    elif target in (
        zt_ops.zentorch_mm_gelu_erf.default,
        zt_ops.zentorch_addmm_gelu_erf.default,
        zt_ops.zentorch_addmm_1dbias_gelu_erf.default,
    ):
        return 3
    else:
        return 0


horizontal_mlp_targets = {
    "mm": [
        zt_ops.zentorch_mm.default,
        zt_ops.zentorch_mm_relu.default,
        zt_ops.zentorch_mm_gelu_tanh.default,
        zt_ops.zentorch_mm_gelu_erf.default,
        zt_ops.zentorch_mm_silu.default,
    ],
    "addmm_1dbias": [
        zt_ops.zentorch_addmm_1dbias.default,
        zt_ops.zentorch_addmm_1dbias_relu.default,
        zt_ops.zentorch_addmm_1dbias_gelu_tanh.default,
        zt_ops.zentorch_addmm_1dbias_gelu_erf.default,
        zt_ops.zentorch_addmm_1dbias_silu.default,
    ],
}


def get_group_attr(target):
    if target in horizontal_mlp_targets["addmm_1dbias"]:
        return {"beta": (1.0, -4), "alpha": (1.0, -3), "is_zentorch_mm": (0, -1)}
    elif target in horizontal_mlp_targets["mm"]:
        return {"beta": (0.0, -4), "alpha": (1.0, -3), "is_zentorch_mm": (1, -1)}
    else:
        return None


def vertical_mlp_fusion(fx_graph):
    vertical_mlp_candidates = {
        zt_ops.zentorch_addmm.default,
        zt_ops.zentorch_addmm_relu.default,
        zt_ops.zentorch_addmm_gelu_tanh.default,
        zt_ops.zentorch_addmm_gelu_erf.default,
        zt_ops.zentorch_addmm_1dbias.default,
        zt_ops.zentorch_addmm_1dbias_relu.default,
        zt_ops.zentorch_addmm_1dbias_gelu_tanh.default,
        zt_ops.zentorch_addmm_1dbias_gelu_erf.default,
    }

    def return_next_addmm(users):
        # GroupMLP fusion is possible only when the consecutive Linear layers
        # have only one output, which must go to the next Linear layer. These
        # two conditions translate to checking if length of users is 1 and the
        # the user is either zentorch_addmm or zentorch_addmm_1dbias. Only when
        # these two conditions are true, True and the next Linear layer node
        # is returned.
        if len(users) == 1 and users[0].target in vertical_mlp_candidates:
            return users[0]
        return None

    logger.info("Fusing vertical contiguous addmm ops.")
    for node in fx_graph.graph.nodes:
        if node.target == at_ops.detach.default and len(node.users) == 0:
            fx_graph.graph.erase_node(node)
    addmm_groups = [[]]
    nodes_traversed = set()
    for node in fx_graph.graph.nodes:
        if node.target in vertical_mlp_candidates:
            while node not in nodes_traversed:
                addmm_groups[-1].append(node)
                nodes_traversed.add(node)
                result_node = return_next_addmm(list(node.users.keys()))
                if not result_node:
                    addmm_groups.append([])
                    break
                node = result_node
    # kwargs in the form (name, default_value, idx_in_args)
    kwargs = [("beta", 1.0, -3), ("alpha", 1.0, -2)]
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
                # Populate "fuse" argument
                group_op_args[-1].append(get_fuse_val(addmm.target))
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
                        "sequence of addmm layers"
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
            last_addmm.target = zt_ops.zentorch_vertical_mlp_group.default
            # Name is form group_mlp_{Op Number}
            if group_idx:
                last_addmm.name = f"group_mlp_{group_idx}"
            else:
                last_addmm.name = "group_mlp"
            group_idx += 1
            for addmm in group[::-1]:
                if addmm != last_addmm:
                    fx_graph.graph.erase_node(addmm)
    fx_graph.recompile()
    return fx_graph


def qkv_fusion(fx_graph):
    # Fusing parallel matmuls in attention block which constitute
    # key query value pair for LLM's.
    logger.info("Detecting and executing QKV parallel ops.")
    groups = {}

    for node in fx_graph.graph.nodes:
        node_name = node.name
        # Pattern check begins with finding the common node
        # for the horizontal matmul pattern
        if len(node.users) >= 3 and node.op != "placeholder":
            for user in node.users.keys():
                # This pattern is legit only if the output tensor of common node acts
                # as input tensor of view ops, should not be used for specifying shape.
                # HF models always have view before a matmul, hence harcoding the index.
                if node is not user.args[0]:
                    continue
                # Skipping if there are no users present or if the user is a return node
                if not bool(user.users.keys()):
                    continue
                user_node = list(user.users)[0]
                # Adding the condition check to find the pattern where
                # the node with number of users > 3 is linked via a
                # convert element and a view to a matmul.
                # Hence, 2 levels of check is required to reach matmul nodes.
                if user_node.target == torch.ops.aten.view.default:
                    user_node = list(user_node.users)[0]

                # Append addmm/mm nodes to group
                # Check if addmm/mm is unique to the dictionary
                node_values = get_group_attr(user_node.target)
                if node_values:
                    groups.setdefault(node_name, {"nodes": []})["nodes"].append(
                        user_node
                    )
    # Perform fusion and optimization
    for group in groups.values():
        # Check for attention block matmuls
        # fuse only the first 3 matmuls: Query, Key, Value
        group["nodes"] = group["nodes"][:3]

        # Validate if all K,Q,V are of same target value
        target_values = [(group["nodes"][i]).target for i in range(len(group["nodes"]))]
        same_target = set(target_values)

        # Checking only for K,Q,V pair hence hardcoded to 3
        if len(same_target) == 1 and len(group["nodes"]) == 3:
            # Checking if the nodes are inter-dependent
            nodes_are_dependent = (
                find_path(fx_graph, group["nodes"][0], group["nodes"][1])
                or find_path(fx_graph, group["nodes"][1], group["nodes"][2])
                or find_path(fx_graph, group["nodes"][0], group["nodes"][2])
            )
            if nodes_are_dependent:
                logger.info("QKV nodes are not independent of each other, cannot perform fusion")
                continue
            group_op_args = [[] for _ in range(7)]
            first_node = group["nodes"][0]

            # Update the group attributes
            node_values = get_group_attr(first_node.target)
            group.update(node_values)

            # Check if node is zentorch_addmm_1d_bias or zentorch_mm.
            # zentorch_mm has only 2 arguments, whereas zentorch_addmm has 3.
            # create an empty node to replicate the arg structure of addmm.
            if len(first_node.args) == 2:
                # take arbitrary shape value for the empty_node being created
                # from the previous node. HF LLMs have view as the previous node.
                # Type checking the arguments for first_node
                shape = first_node.args[0].args[1]
                if isinstance(shape, (list, tuple)):
                    empty_shape = [list(shape)]
                elif isinstance(shape, int):
                    empty_shape = [[shape]]
                else:
                    raise TypeError(
                        f"Unexpected type for shape: {type(shape)}. "
                        "Expected list, tuple, or int."
                    )
                empty_node_args = tuple(empty_shape)
                empty_node = fx_graph.graph.create_node(
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
                # iterate through kwargs to append node's value, if present in graph,
                # else append the default value
                for kwarg in ["alpha", "beta", "is_zentorch_mm"]:
                    if group[kwarg] in node.kwargs:
                        group_op_args[group[kwarg][1]].append(node.kwargs[kwarg])
                    else:
                        group_op_args[group[kwarg][1]].append(group[kwarg][0])
                # Update fuse values.
                group_op_args[5].append(get_fuse_val(node.target))
            # Create zentorch_attn_qkv_fusion node
            group_node = fx_graph.graph.create_node(
                op="call_function",
                target=zt_ops.zentorch_attn_qkv_fusion.default,
                args=tuple(group_op_args),
            )
            first_node.prepend(group_node)

            # Incrementing the fusion counter
            counters["zentorch"]["qkv_fusion"] += 1

            # Creating getitem nodes to parse the output vector.
            getitem_nodes = []
            for getitem_num in range(3):
                new_node = fx_graph.graph.create_node(
                    op="call_function",
                    target=operator.getitem,
                    args=(group_node, getitem_num),
                )
                group_node.append(new_node)  # FX API
                getitem_nodes.append(new_node)  # LIST API
            for i, node in enumerate(group["nodes"]):
                node.replace_all_uses_with(getitem_nodes[i])
                fx_graph.graph.erase_node(node)

            # sort graph topologically
            stable_topological_sort(fx_graph.graph)
            fx_graph.graph.lint()
        else:
            logger.info(
                "Horizontal Group fusion not possible with the current \
                            combination of addmm/mm layers"
            )

    fx_graph.recompile()
    return fx_graph


def eb_group_mlp_group_fusion(fx_graph):
    logger.info(
        "Fusing the horizontally fused EmbeddingBag op and the "
        "vertically fused MLP op"
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
            if node_input.target == zt_ops.zentorch_vertical_mlp_group.default:
                group_mlp_op = node_input
            # Here we strictly checking that every get_item node has the
            # horizontally fused EmbeddingBag op as the input. If that is not
            # the case, the fusion does not take place and the control goes to
            # the next interaction node.
            elif node_input.target == operator.getitem:
                for getitem_input in node_input.all_input_nodes:
                    condition = (
                        getitem_input.target
                        == zt_ops.zentorch_horizontal_embedding_bag_group.default
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
                logger.info(
                    "Fusion of horizontally fused EmbeddingBag op and "
                    "the vertically fused MLP op into one single op is "
                    "not possible at the current concate node: %s!",
                    node,
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
            group_mlp_op.target = zt_ops.zentorch_fused_eb_mlp.default
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

    fx_graph.recompile()
    return fx_graph
