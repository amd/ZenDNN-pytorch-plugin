# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Automatic linear fusion for dense multi-gate mixture-of-experts graphs.

Recognize task-specific softmax gates mixing the same stacked expert outputs.
No model class names, module metadata, or caller opt-in are required.
"""

import operator

import torch
from torch._inductor.pattern_matcher import stable_topological_sort

from ._custom_op_replacement import needs_contiguous_for_node
from ._logging import get_logger
from ._utils import counters

logger = get_logger(__name__)
at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch


def _is(node, target):
    return isinstance(node, torch.fx.Node) and node.target == target


def _meta(node):
    value = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return value if isinstance(value, torch.Tensor) else None


def _strip_cast(node):
    while isinstance(node, torch.fx.Node) and node.target in (
        torch.ops.prims.convert_element_type.default,
        at_ops._to_copy.default,
    ):
        node = node.args[0]
    return node


def _reduces(node, target, dims, keepdim=False):
    if not _is(node, target) or len(node.args) < 2:
        return False
    actual_keepdim = (
        node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
    )
    return list(node.args[1]) in dims and actual_keepdim == keepdim


def _gate_linear(weights):
    """Match a last-axis softmax of a gate linear, including decomposition."""
    weights = _strip_cast(weights)
    if weights.target in (at_ops._softmax.default, at_ops.softmax.int):
        if weights.args[1] not in (-1, 1):
            return None
        logits = weights.args[0]
    elif _is(weights, at_ops.div.Tensor):
        exp, denom = weights.args[:2]
        if (
            not _is(exp, at_ops.exp.default)
            or not _reduces(denom, at_ops.sum.dim_IntList, [[-1], [1]], True)
            or denom.args[0] is not exp
        ):
            return None
        sub = exp.args[0]
        if not _is(sub, at_ops.sub.Tensor):
            return None
        logits, maximum = sub.args[:2]
        if not _reduces(maximum, at_ops.amax.default, [[-1], [1]], True):
            return None
        if maximum.args[0] is not logits:
            return None
    else:
        return None
    logits = _strip_cast(logits)
    if not _is(logits, zt_ops.zentorch_linear_unary.default):
        return None
    return logits if logits.kwargs.get("post_op", "none") == "none" else None


def _expert_outputs(stacked):
    """Recognize stack(experts, 1) and its cat/view decompositions."""
    value = _meta(stacked)
    if value is None or value.ndim != 3:
        return None
    count = value.shape[1]
    if not isinstance(count, int) or count < 2:
        return None
    if _is(stacked, at_ops.stack.default) and stacked.args[1] in (1, -2):
        outputs = list(stacked.args[0])
    else:
        cat = stacked
        if stacked.target in (at_ops.view.default, at_ops.reshape.default):
            cat = stacked.args[0]
        if not _is(cat, at_ops.cat.default) or cat.args[1] != 1:
            return None
        outputs = list(cat.args[0])
        if all(
            _is(n, at_ops.unsqueeze.default) and n.args[1] in (1, -2) for n in outputs
        ):
            outputs = [n.args[0] for n in outputs]
    if len(outputs) != count:
        return None
    for output in outputs:
        meta = _meta(output)
        if (
            meta is None
            or meta.ndim != 2
            or meta.shape != (value.shape[0], value.shape[2])
        ):
            return None
    return outputs


def _ancestors_until(output, boundary):
    seen = set()
    pending = [output]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if node is not boundary:
            pending.extend(node.all_input_nodes)
    return seen


def _mixture_operands(node):
    if _reduces(node, at_ops.sum.dim_IntList, [[1], [-2]]):
        product = _strip_cast(node.args[0])
        if not _is(product, at_ops.mul.Tensor):
            return
        for stacked, weights in (product.args[:2], product.args[:2][::-1]):
            if _is(weights, at_ops.unsqueeze.default) and weights.args[1] in (-1, 2):
                yield stacked, weights.args[0]
    elif node.target in (at_ops.bmm.default, zt_ops.zentorch_bmm.default):
        weights, stacked = node.args[:2]
        if _is(weights, at_ops.unsqueeze.default) and weights.args[1] in (1, -2):
            yield stacked, weights.args[0]


def _mmoe_regions(graph):
    """Require at least two distinct gates mixing one shared expert stack."""
    mixtures = {}
    for node in graph.nodes:
        for stacked, weights in _mixture_operands(node):
            outputs = _expert_outputs(stacked)
            if outputs is None:
                continue
            gate = _gate_linear(weights)
            if gate is None:
                continue
            gate_meta = _meta(gate)
            stack_meta = _meta(stacked)
            if gate_meta is None or gate_meta.shape != stack_meta.shape[:2]:
                continue
            mixtures.setdefault((stacked, gate.args[0]), set()).add(gate)

    for (stacked, shared_input), gates in mixtures.items():
        if len(gates) < 2:
            continue
        expert_nodes = [
            _ancestors_until(output, shared_input)
            for output in _expert_outputs(stacked)
        ]
        if not all(shared_input in nodes for nodes in expert_nodes):
            continue
        region = set().union(*expert_nodes) - {shared_input}
        region.update(gates)
        yield {n for n in region if _is(n, zt_ops.zentorch_linear_unary.default)}


def _binary_consumers(node):
    pending = [(node, 0)]
    found = set()
    unary = (at_ops.silu, at_ops.relu, at_ops.gelu, at_ops.sigmoid, at_ops.tanh)
    while pending:
        node, depth = pending.pop()
        for user in node.users:
            packet = getattr(user.target, "overloadpacket", None)
            if packet in (at_ops.add, at_ops.mul):
                found.add(user)
            elif depth < 2 and packet in unary:
                pending.append((user, depth + 1))
    return found


def _fusion_reduces_traffic(members, weight_idx):
    seen = set()
    displaces = False
    for member in members:
        consumers = _binary_consumers(member)
        displaces |= bool(seen.intersection(consumers))
        seen.update(consumers)
    if not displaces:
        return True
    # Rough tensor-traffic estimate: assume separate linears each read the
    # input, so grouping saves N-1 reads. Charge every output when a binary
    # post-op is displaced; for a pair this reduces to 2H < K. Larger groups
    # can pass even when individual pairs do not, because they save more reads.
    # This is not a latency bound: cache reuse and GEMM kernel costs are omitted.
    weights = [_meta(n.args[weight_idx]) for n in members]
    return sum(w.shape[0] for w in weights) < (len(members) - 1) * weights[0].shape[1]


# Minimum number of parallel linear nodes that a group must have to be fused.
PARALLEL_LINEAR_FUSION_MIN_GROUP = 2


def group_has_internal_dependency(group):
    """True if any node of the group is reachable from another node of it.

    Such nodes are not computed in parallel, so folding them into one GEMM
    would change the order of the graph.
    """
    group_set = set(group)
    for start_node in group:
        stack = list(start_node.users)
        visited = set()
        while stack:
            current_node = stack.pop()
            if current_node in visited:
                continue
            visited.add(current_node)
            if current_node in group_set:
                return True
            stack.extend(
                user_node
                for user_node in current_node.users
                if user_node not in visited
            )
    return False


def fusable_signature(node, weight_idx, bias_idx):
    """Key describing what a linear node can be fused with, None if unfusable.

    Nodes only share a GEMM when the fused weight is a valid cat of theirs and
    the single post-op of the fused node applies to every output, so the key
    carries everything the fused node cannot vary per member.
    """
    # cat on prepacked weight tensors does not produce a valid weight
    if node.kwargs.get("is_weight_prepacked", False):
        logger.info("Skipping fusion: linear node has prepacked weights")
        return None
    weight_meta = node.args[weight_idx].meta.get("val", None)
    if weight_meta is None:
        return None
    # Bias usage is a compatibility constraint, not an expert/gate role tag.
    # Either role may have or omit bias; matching signatures can combine both.
    # We do not synthesize zero biases or assume this partition is the fastest:
    # the trade-off depends on shape, dtype and the backend's GEMM kernels.
    has_bias = len(node.args) > bias_idx and node.args[bias_idx] is not None
    return (
        has_bias,
        node.kwargs.get("post_op", "none"),
        weight_meta.dtype,
        weight_meta.shape[1],
    )


def collect_parallel_linear_groups(node, eligible):
    """Group compatible linears inside one recognized MMoE region."""
    input_node = node.args[0]
    input_users = [
        user
        for user in input_node.users
        if user in eligible and user.target == zt_ops.zentorch_linear_unary.default
    ]
    if len(input_users) < PARALLEL_LINEAR_FUSION_MIN_GROUP:
        return []

    node_schema = input_users[0].target._schema
    arg_indices = {arg.name: i for i, arg in enumerate(node_schema.arguments)}
    bias_idx = arg_indices["bias"]
    weight_idx = arg_indices["weight"]

    buckets = {}
    for user in input_users:
        signature = fusable_signature(user, weight_idx, bias_idx)
        if signature is None:
            continue
        buckets.setdefault(signature, []).append(user)

    groups = []
    for signature, members in buckets.items():
        if len(members) < PARALLEL_LINEAR_FUSION_MIN_GROUP:
            continue
        # Check that nodes are independent (no dependencies between them)
        if group_has_internal_dependency(members):
            logger.info("Parallel linear nodes are dependent, skipping fusion")
            continue
        if not _fusion_reduces_traffic(members, weight_idx):
            continue
        logger.info(
            "Fusing %d parallel linear nodes with signature %s", len(members), signature
        )
        groups.append(members)
    return groups


def mmoe_fusion(fx_graph):
    """Fuse linears only in structurally recognized multi-gate expert regions."""
    groups = []
    claimed = set()
    for eligible in _mmoe_regions(fx_graph):
        visited = set()
        for node in fx_graph.nodes:
            if node in eligible and node not in visited and node not in claimed:
                new_groups = collect_parallel_linear_groups(node, eligible - claimed)
                visited.update(node.args[0].users)
                groups.extend(new_groups)
                claimed.update(member for group in new_groups for member in group)

    # Loop over the groups collected for fusion. Concatenate the weights and biases
    # (if any) of the linear nodes in each group. Create a new linear node with the
    # same input but fused (concatenated) weight and bias. Split the fused result
    # back into the original outputs and remove the original linear nodes.
    nodes_to_remove = []
    for group in groups:
        input_tensor = group[0].args[0]

        node_schema = group[0].target._schema
        arg_indices = {arg.name: i for i, arg in enumerate(node_schema.arguments)}
        weight_idx = arg_indices["weight"]
        bias_idx = arg_indices["bias"]

        # Output sizes come from the original weights, they may differ per node.
        split_sections = [node.args[weight_idx].meta["val"].shape[0] for node in group]
        has_bias = len(group[0].args) > bias_idx and group[0].args[bias_idx] is not None

        # Create concatenated weight: cat([W0, W1, ...], dim=0)
        with fx_graph.inserting_after(group[-1]):
            fused_weight = fx_graph.create_node(
                op="call_function",
                target=at_ops.cat.default,
                args=([node.args[weight_idx] for node in group], 0),
            )

            # Create concatenated bias if it exists: cat([b0, b1, ...], dim=0)
            fused_bias = None
            if has_bias:
                fused_bias = fx_graph.create_node(
                    op="call_function",
                    target=at_ops.cat.default,
                    args=([node.args[bias_idx] for node in group], 0),
                )

        # Create the fused linear operation: X @ W_fused + b_fused
        # For the kwargs, we can use the first node's kwargs directly since
        # we have checks to ensure that the post-ops are consistent
        # and weights are not prepacked.
        with fx_graph.inserting_after(fused_bias if fused_bias else fused_weight):
            fused_node = fx_graph.create_node(
                op="call_function",
                target=zt_ops.zentorch_linear_unary.default,
                args=(input_tensor, fused_weight, fused_bias),
                kwargs=group[0].kwargs,
            )

        counters["zentorch"]["mmoe_fusion_linear"] += 1
        counters["zentorch"]["mmoe_fusion_members"] += len(group)

        # Split the fused output back into the original per-node outputs
        with fx_graph.inserting_after(fused_node):
            split_node = fx_graph.create_node(
                op="call_function",
                target=at_ops.split_with_sizes.default,
                args=(fused_node, split_sections, -1),
            )

        # Create getitem nodes to extract the individual tensors
        for idx, original_node in enumerate(group):
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
                counters["zentorch"]["mmoe_fusion_linear_contiguous"] += 1
            else:
                # Replace all uses of the original node with the split output
                original_node.replace_all_uses_with(getitem_node)

            nodes_to_remove.append(original_node)

    # Clean up: Remove the original linear nodes
    for node in nodes_to_remove:
        fx_graph.erase_node(node)

    if len(nodes_to_remove) > 0:
        stable_topological_sort(fx_graph)
        fx_graph.lint()

    return fx_graph
