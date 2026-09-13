# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from torch._inductor.pattern_matcher import stable_topological_sort
from torch._inductor.utils import is_view
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
                # This fusion rewrites producers to op.out(out_slice, *args)
                # (out FIRST). Skip ops whose .out takes out last (aten
                # convention, e.g. dynamic_qlinear.out) to avoid mis-mapping.
                first_arg = op.out._schema.arguments[0]
                if first_arg.alias_info is None or not first_arg.alias_info.is_write:
                    logger.info(
                        "Skipping cat-fold for %s (.out takes out as a "
                        "non-leading arg)",
                        arg.target._opname,
                    )
                    return False
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


def inplace_cat_fusion(fx_graph):
    nodes_to_remove = []
    for node in fx_graph.nodes:
        if node.target == torch.ops.aten.cat.default and has_out_variant_for_all_args(
            node
        ):
            if not is_cat_dim_valid_for_folding(node):
                logger.info("Cat node %s dim is not valid for fusion", node)
                continue

            # create the output node
            shape = node.meta["tensor_meta"].shape
            with fx_graph.inserting_after(node):
                get_out_node = fx_graph.create_node(
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
                with fx_graph.inserting_after(get_out_node):
                    as_strided_node = fx_graph.create_node(
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
                with fx_graph.inserting_after(as_strided_node):
                    new_args = (
                        as_strided_node,
                    ) + arg.args  # Correctly construct the new_args tuple
                    out_node = fx_graph.create_node(
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
        fx_graph.erase_node(node)

    if len(nodes_to_remove) > 0:
        stable_topological_sort(fx_graph)
        fx_graph.lint()
    return fx_graph


def collect_grouped_emb_bag_args(group_op, nodes):
    # Extract the schema
    # This function assumes that individual node and grouped node
    # have the corresponding arguments in the same position.
    # This function only handles args, kwargs are not handled.
    # If there are multiple args with default values, and
    # the user provided any one in the middle, torch fills the rest
    # of args with the default values.
    schema = group_op._schema
    node_schema = nodes[0].target._schema
    grouped_args = []
    for i, arg in enumerate(schema.arguments):
        typ = str(arg.type)
        if "List" in typ:
            collected = []
            for node in nodes:
                if i < len(node.args):
                    collected.append(node.args[i])
                elif node_schema.arguments[i].default_value is not None:
                    collected.append(node_schema.arguments[i].default_value)
                else:
                    collected.append(None)
            grouped_args.append(collected)
        else:
            if i < len(nodes[0].args):
                grouped_args.append(nodes[0].args[i])

    return tuple(grouped_args)


def emb_ops_horizontal_fusion(fx_graph):
    """
    Fuse horizontal parallel embedding operations into group operations.

    This function identifies consecutive embedding operations of the same type
    and fuses them into group operations for better performance. For example,
    multiple zentorch_embedding_bag operations will be fused into a single
    zentorch_horizontal_embedding_bag_group operation.

    Args:
        fx_graph: FX graph to optimize

    Returns:
        fx_graph: Optimized FX graph with fused embedding operations
    """
    logger.info("Fusing horizontal parallel embedding ops.")

    # Mapping from individual embedding ops to their corresponding group ops
    zentorch_embed_ops_dict = {
        zt_ops.zentorch_embedding_bag.default: zt_ops.zentorch_horizontal_embedding_bag_group.default,
        zt_ops.zentorch_embedding.default: zt_ops.zentorch_horizontal_embedding_group.default,
        zt_ops.zentorch_quant_embedding_bag.default: zt_ops.zentorch_horizontal_quant_embedding_bag_group.default,
        zt_ops.zentorch_quant_embedding_bag.out: zt_ops.zentorch_horizontal_quant_embedding_bag_group.out,
    }

    # Storage for all groups found in the graph
    groups = {}
    # Storage for the current group being built
    current_group = {}
    # Track users of nodes in current group to detect group boundaries
    users_of_current_group = []
    for node in fx_graph.nodes:
        if node.target in zentorch_embed_ops_dict:
            if node.target not in groups:
                groups[node.target] = []
            if node.target not in current_group:
                current_group[node.target] = []

            # Add node to current group and track its users
            current_group[node.target].append(node)
            users_of_current_group.extend(node.users)

        # If we hit a user of any node in current group, finalize the group
        elif node in users_of_current_group:
            # Move all current groups to the main groups storage
            for op, nodes in current_group.items():
                groups[op].append(nodes)
            # Reset for next group
            current_group = {}
            users_of_current_group = []

    # Handle any remaining groups at the end of the graph
    for op, nodes in current_group.items():
        groups[op].append(nodes)

    # Replace individual operations with group operations
    # groups[emb] = [[node1, node2], [node3, node4]]
    nodes_to_remove = []
    for op, node_groups in groups.items():
        for nodes in node_groups:
            if len(nodes) < 2:
                continue  # No fusion needed for single nodes

            # Get the target group operation for this embedding type
            group_target = zentorch_embed_ops_dict[op]

            # Collect and restructure arguments according to group op schema
            group_args = collect_grouped_emb_bag_args(group_target, nodes)

            # Create the group operation node
            with fx_graph.inserting_after(nodes[-1]):
                group_node = fx_graph.create_node(
                    op="call_function",
                    target=group_target,
                    args=group_args,
                )

            counters["zentorch"][group_target._opname] += 1

            # Mark original nodes for removal
            nodes_to_remove.extend(nodes)

            # Group operations return lists/tuples, so we need getitem nodes
            # to extract individual results for each original operation
            if ".out" not in group_target.__name__:
                for idx, node in enumerate(nodes):
                    with fx_graph.inserting_after(group_node):
                        # Create getitem node to extract result at index idx
                        getitem_node = fx_graph.create_node(
                            op="call_function",
                            target=operator.getitem,
                            args=(group_node, idx),
                        )
                    # Replace all uses of original node with the getitem result
                    node.replace_all_uses_with(getitem_node)

    # Clean up: Remove all original embedding nodes that were fused
    for node in nodes_to_remove:
        fx_graph.erase_node(node)

    if len(nodes_to_remove) > 0:
        stable_topological_sort(fx_graph)
        fx_graph.lint()
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
    for node in fx_graph.nodes:
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
                # Use schema to get argument indices for both nodes
                pred_schema = pred_node.target._schema
                pred_arg_indices = {
                    arg.name: i for i, arg in enumerate(pred_schema.arguments)
                }
                curr_schema = curr_node.target._schema
                curr_arg_indices = {
                    arg.name: i for i, arg in enumerate(curr_schema.arguments)
                }

                # Get predecessor's output_dtype from positional args
                pred_output_dtype_idx = pred_arg_indices["output_dtype"]
                pred_output_scales_idx = pred_arg_indices["output_scales"]
                pred_output_zp_idx = pred_arg_indices["output_zero_points"]
                pred_args = list(pred_node.args)
                pred_output_dtype = (
                    pred_args[pred_output_dtype_idx]
                    if pred_output_dtype_idx < len(pred_args)
                    else None
                )
                pred_output_scales = (
                    pred_args[pred_output_scales_idx]
                    if pred_output_scales_idx < len(pred_args)
                    else None
                )
                pred_output_zp = (
                    pred_args[pred_output_zp_idx]
                    if pred_output_zp_idx < len(pred_args)
                    else None
                )
                # if dtype is not None or
                # if dtype is None then o/p-scales & o/p-zp should also
                # be None and copy all the arguments, then below holds good
                if (
                    pred_output_dtype is None
                    and pred_output_scales is None
                    and pred_output_zp is None
                ) or (
                    pred_output_dtype is not None
                    and pred_output_dtype
                    in (
                        torch.float,
                        torch.bfloat16,
                    )
                ):
                    input_scales_idx = curr_arg_indices["input_scales"]
                    input_zp_idx = curr_arg_indices["input_zero_points"]
                    new_output_dtype = (
                        torch.int8 if curr_args[input_zp_idx] is None else torch.uint8
                    )
                    pred_args = pred_args[:-3] + [
                        curr_args[input_scales_idx],
                        curr_args[input_zp_idx],
                        new_output_dtype,
                    ]
                    pred_node.args = tuple(pred_args)
                    counters["zentorch"]["optimized_reorder"] += 1
                pred_node = curr_node
    stable_topological_sort(fx_graph)
    fx_graph.lint()
    return fx_graph


def needs_contiguous_for_node(start_node):
    """
    Traverse users starting from start_node.
    Returns True if as_strided is found before any non-view op.
    Returns False if non-view op is found first.
    """
    # DFS through users
    stack = [start_node]
    visited = set()

    while len(stack) > 0:
        current_node = stack.pop()

        if current_node in visited:
            continue
        visited.add(current_node)

        # Loop through all users
        for user_node in current_node.users:
            if user_node in visited:
                continue

            # Check if user is a non-view op - if yes, no contiguous needed
            if (
                user_node.op == "call_function"
                and isinstance(user_node.target, torch._ops.OpOverload)
                and not is_view(user_node.target)
            ):
                continue  # Skip this path, but continue checking other paths

            # Check if user is as_strided - if yes, need contiguous
            if user_node.target == at_ops.as_strided.default:
                return True

            # Otherwise, continue traversing
            stack.append(user_node)

    return False


# qkv_fusion pass with zentorch linear ops
def qkv_fusion(fx_graph):
    logger.info("Fusing QKV parallel linear operations.")

    qkv_fusion_len = 3
    groups = []
    nodes_visited = set()

    # Loop over the graph to find groups of QKV nodes (3 linear nodes with the same input)
    for node in fx_graph.nodes:
        if (
            node.target == zt_ops.zentorch_linear_unary.default
            and node not in nodes_visited
        ):
            input_node = node.args[0]
            users = list(input_node.users.keys())
            input_users = []
            for user in users:
                if user.target == zt_ops.zentorch_linear_unary.default:
                    input_users.append(user)
            # Check if this input has exactly 3 users (Q, K, V)
            if len(input_users) != qkv_fusion_len:
                logger.info(
                    "Fusion only supported for exactly 3 linear nodes currently, skipping fusion"
                )
                continue
            # Check that nodes are independent (no dependencies between them)
            nodes_are_dependent = (
                find_path(fx_graph, input_users[0], input_users[1])
                or find_path(fx_graph, input_users[1], input_users[2])
                or find_path(fx_graph, input_users[0], input_users[2])
            )
            if nodes_are_dependent:
                logger.info("QKV nodes are dependent, skipping fusion")
                continue

            # Check that all nodes have consistent bias usage (all have bias or none have bias)
            q_node_schema = input_users[0].target._schema
            arg_indices = {arg.name: i for i, arg in enumerate(q_node_schema.arguments)}
            bias_idx = arg_indices["bias"]

            q_has_bias = (
                len(input_users[0].args) > bias_idx
                and input_users[0].args[bias_idx] is not None
            )
            k_has_bias = (
                len(input_users[1].args) > bias_idx
                and input_users[1].args[bias_idx] is not None
            )
            v_has_bias = (
                len(input_users[2].args) > bias_idx
                and input_users[2].args[bias_idx] is not None
            )

            if not (q_has_bias == k_has_bias == v_has_bias):
                logger.info("QKV nodes have inconsistent bias usage, skipping fusion")
                continue

            # Skip fusion if any linear has prepacked weights since
            # cat on prepacked weight tensors does not produce a valid weight
            any_prepacked = any(
                user.kwargs.get("is_weight_prepacked", False) for user in input_users
            )
            if any_prepacked:
                logger.info(
                    "Skipping fusion: one or more linear nodes have prepacked weights"
                )
                continue

            # Skip fusion if linears have different post-ops since the fused
            # linear can only apply a single post-op to the entire output
            post_ops = {user.kwargs.get("post_op", "none") for user in input_users}
            if len(post_ops) > 1:
                logger.info(
                    "Skipping fusion: linear nodes have inconsistent post-ops %s",
                    post_ops,
                )
                continue

            groups.append(input_users)
            nodes_visited.update(input_users)

    # Loop over the QKV groups collected for fusion. Concatenate the weights and biases(if any) of the three linear nodes
    # in each group. Create a new linear node with same input but fused(concatenated) weight and bias.
    # Split the fused node result back into QKV nodes and remove the original linear nodes.
    nodes_to_remove = []
    for group in groups:
        # Extract Q, K, V nodes
        q_node, k_node, v_node = group[0], group[1], group[2]
        input_tensor = q_node.args[0]

        q_node_schema = q_node.target._schema
        arg_indices = {arg.name: i for i, arg in enumerate(q_node_schema.arguments)}
        weight_idx = arg_indices["weight"]
        bias_idx = arg_indices["bias"]

        # Get metadata from original weight tensors
        q_weight_meta = q_node.args[weight_idx].meta["val"]
        k_weight_meta = k_node.args[weight_idx].meta["val"]
        v_weight_meta = v_node.args[weight_idx].meta["val"]

        # Create concatenated weight: cat([Wq, Wk, Wv], dim=0)
        with fx_graph.inserting_after(v_node):
            fused_weight = fx_graph.create_node(
                op="call_function",
                target=at_ops.cat.default,
                args=(
                    [
                        q_node.args[weight_idx],
                        k_node.args[weight_idx],
                        v_node.args[weight_idx],
                    ],
                    0,
                ),
            )

            # Create concatenated bias if it exists: cat([bq, bk, bv], dim=0)
            fused_bias = None
            if len(q_node.args) > bias_idx and q_node.args[bias_idx] is not None:
                fused_bias = fx_graph.create_node(
                    op="call_function",
                    target=at_ops.cat.default,
                    args=(
                        [
                            q_node.args[bias_idx],
                            k_node.args[bias_idx],
                            v_node.args[bias_idx],
                        ],
                        0,
                    ),
                )

        # Create fused QKV linear operation: X @ Wqkv + bqkv
        # For the kwargs, we can use q_node.kwargs directly since
        # we have checks to ensure that the post-ops are consistent
        # and weights are not prepacked.
        qkv_node_args = [input_tensor, fused_weight, fused_bias]
        with fx_graph.inserting_after(fused_bias if fused_bias else fused_weight):
            qkv_fused_node = fx_graph.create_node(
                op="call_function",
                target=zt_ops.zentorch_linear_unary.default,
                args=tuple(qkv_node_args),
                kwargs=q_node.kwargs,
            )

        counters["zentorch"]["qkv_fusion_linear"] += 1

        # Split the fused output back into Q, K, V
        # Get the output dimensions from the original weight shapes
        q_output_dim = q_weight_meta.shape[0]
        k_output_dim = k_weight_meta.shape[0]
        v_output_dim = v_weight_meta.shape[0]

        # Create split nodes to extract Q, K, V from the fused output
        split_sections = [q_output_dim, k_output_dim, v_output_dim]
        with fx_graph.inserting_after(qkv_fused_node):
            split_node = fx_graph.create_node(
                op="call_function",
                target=at_ops.split_with_sizes.default,
                args=(qkv_fused_node, split_sections, -1),
            )

        # Create getitem nodes to extract individual Q, K, V tensors
        for idx, original_node in enumerate([q_node, k_node, v_node]):
            with fx_graph.inserting_after(split_node):
                getitem_node = fx_graph.create_node(
                    op="call_function",
                    target=operator.getitem,
                    args=(split_node, idx),
                )

            # If as_strided is found downstream with only view ops, add a contiguous call
            needs_contiguous = needs_contiguous_for_node(original_node)

            if needs_contiguous:
                with fx_graph.inserting_after(getitem_node):
                    contiguous_node = fx_graph.create_node(
                        op="call_function",
                        target=at_ops.contiguous.default,
                        args=(getitem_node,),
                    )
                    # Replace all uses of the original node with contiguous output
                    original_node.replace_all_uses_with(contiguous_node)
                counters["zentorch"]["qkv_fusion_linear_contiguous"] += 1
            else:
                # Replace all uses of the original node with the split output
                original_node.replace_all_uses_with(getitem_node)

            nodes_to_remove.append(original_node)

    # Clean up: Remove original Q, K, V nodes
    for node in nodes_to_remove:
        fx_graph.erase_node(node)

    if len(nodes_to_remove) > 0:
        stable_topological_sort(fx_graph)
        fx_graph.lint()

    return fx_graph
