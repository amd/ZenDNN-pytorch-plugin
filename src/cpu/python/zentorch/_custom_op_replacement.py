# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from torch._inductor.pattern_matcher import stable_topological_sort
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
        # zt_ops.zentorch_quant_embedding_bag.default: zt_ops.zentorch_horizontal_quant_embedding_bag_group.default,
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
    elif target in (
        zt_ops.zentorch_mm_silu.default,
        zt_ops.zentorch_addmm_silu.default,
        zt_ops.zentorch_addmm_1dbias_silu.default,
    ):
        return 4
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
        zt_ops.zentorch_addmm_silu.default,
        zt_ops.zentorch_addmm_1dbias_relu.default,
        zt_ops.zentorch_addmm_1dbias_gelu_tanh.default,
        zt_ops.zentorch_addmm_1dbias_gelu_erf.default,
        zt_ops.zentorch_addmm_1dbias_silu.default,
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
    stable_topological_sort(fx_graph)
    fx_graph.lint()
    return fx_graph


def qkv_fusion(fx_graph):
    # Fusing parallel matmuls in attention block which constitute
    # key query value pair for LLM's.
    logger.info("Detecting and executing QKV parallel ops.")
    groups = {}

    for node in fx_graph.nodes:
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
                logger.info(
                    "QKV nodes are not independent of each other, cannot perform fusion"
                )
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
                empty_node = fx_graph.create_node(
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
            group_node = fx_graph.create_node(
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
                new_node = fx_graph.create_node(
                    op="call_function",
                    target=operator.getitem,
                    args=(group_node, getitem_num),
                )
                group_node.append(new_node)  # FX API
                getitem_nodes.append(new_node)  # LIST API
            for i, node in enumerate(group["nodes"]):
                node.replace_all_uses_with(getitem_nodes[i])
                fx_graph.erase_node(node)

            # sort graph topologically
            stable_topological_sort(fx_graph)
            fx_graph.lint()
        else:
            logger.info(
                "Horizontal Group fusion not possible with the current \
                            combination of addmm/mm layers"
            )

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
    for node in fx_graph.nodes:
        if node.target == torch.ops.aten.cat.default:
            concat_nodes.append(node)

    # Return the index of the required_node in the fx_graph. Since the fx_graph
    # is a doubly linked list, the index of the required_node in a graph makes
    # sense.
    def get_node_index_in_graph(required_node, fx_graph):
        for idx, node in enumerate(fx_graph.nodes):
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
            new_node = fx_graph.create_node(
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
            fx_graph.erase_node(group_eb_op)
    stable_topological_sort(fx_graph)
    fx_graph.lint()
    return fx_graph
