# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import operator
import sys
import torch
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    register_graph_pattern,
    CallFunction,
    KeywordArg,
    Arg,
    Match,
)
from torch._inductor import config
import functools
from ._utils import counters
from .utils import is_zendnn_embedding_bag_supported

# import the custom logging module
from ._logging import get_logger

at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch

# make a logger for this file
logger = get_logger(__name__)

# pass patterns for the graph
pass_pattern = PatternMatcherPass()


# we will write checks and register_graph_patterns for the following ops
# embedding replacement
def is_embedding_op_replacable(match):
    return match.args[1].meta["val"].ndim == 1


@register_graph_pattern(
    CallFunction(
        at_ops.embedding,
        Arg(),
        Arg(),
        padding_idx=KeywordArg("padding_idx"),
        scale_grad_by_freq=KeywordArg("scale_grad_by_freq"),
        sparse=KeywordArg("sparse"),
    ),
    pass_dict=pass_pattern,
    extra_check=is_embedding_op_replacable,
)
def embedding_replacement(
    match, weight, indices, padding_idx, scale_grad_by_freq, sparse
):
    def repl(weight, indices, padding_idx, scale_grad_by_freq, sparse):
        counters["zentorch"]["zentorch_embedding"] += 1
        return zt_ops.zentorch_embedding(
            weight,
            indices,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

    match.replace_by_example(
        repl, [weight, indices, padding_idx, scale_grad_by_freq, sparse]
    )


# mm replacement
@register_graph_pattern(
    CallFunction(at_ops.mm, Arg(), Arg()),
    pass_dict=pass_pattern,
)
def mm_replacement(match, mat_1, mat_2):
    def repl(mat_1, mat_2):
        counters["zentorch"]["zentorch_mm"] += 1
        return zt_ops.zentorch_mm(mat_1, mat_2)

    match.replace_by_example(repl, [mat_1, mat_2])


# bmm replacement
@register_graph_pattern(
    CallFunction(at_ops.bmm, Arg(), Arg()),
    pass_dict=pass_pattern,
)
def bmm_replacement(match, mat_1, mat_2):
    def repl(mat_1, mat_2):
        counters["zentorch"]["zentorch_bmm"] += 1
        return zt_ops.zentorch_bmm(mat_1, mat_2)

    match.replace_by_example(repl, [mat_1, mat_2])


# addmm->addmm_1dbias replacement
def is_bias_1d_tensor(match):
    # checks if self/bias tensor is 1-d or not
    # returns true if 1d bias tensor
    return match.args[0].meta["val"].ndim == 1


@register_graph_pattern(
    CallFunction(
        at_ops.addmm,
        Arg(),
        Arg(),
        Arg(),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    pass_dict=pass_pattern,
    extra_check=is_bias_1d_tensor,
)
def addmm_1dbias_replacement(match, inp, mat_1, mat_2, *, beta, alpha):
    def repl(inp, mat_1, mat_2, beta, alpha):
        counters["zentorch"]["zentorch_addmm_1dbias"] += 1
        return zt_ops.zentorch_addmm_1dbias(inp, mat_1, mat_2, beta=beta, alpha=alpha)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha])


# addmm replacement
def is_bias_not_1d_tensor(match):
    # checks if self/bias tensor is 1-d or not
    # returns true if 1d bias tensor
    return not is_bias_1d_tensor(match)


@register_graph_pattern(
    CallFunction(
        at_ops.addmm,
        Arg(),
        Arg(),
        Arg(),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    pass_dict=pass_pattern,
    extra_check=is_bias_not_1d_tensor,
)
def addmm_replacement(match, inp, mat_1, mat_2, *, beta, alpha):
    def repl(inp, mat_1, mat_2, beta, alpha):
        counters["zentorch"]["zentorch_addmm"] += 1
        return zt_ops.zentorch_addmm(inp, mat_1, mat_2, beta=beta, alpha=alpha)

    match.replace_by_example(repl, [inp, mat_1, mat_2, beta, alpha])


# baddbmm replacement
def is_baddbmm_replacable(match):
    # TODO: Below shape checks needs to be removed
    # ones the broadcast support for zentorch_baddbmm
    # op is added.
    add_shape = match.args[0].meta["val"].size()
    mat1_shape = match.args[1].meta["val"].size()
    mat2_shape = match.args[2].meta["val"].size()
    shape_check = (
        add_shape[0] == mat1_shape[0]
        and add_shape[1] == mat1_shape[1]
        and add_shape[2] == mat2_shape[2]
    )
    return all(match.args[i].meta["val"].ndim == 3 for i in range(0, 3)) and shape_check


@register_graph_pattern(
    CallFunction(
        at_ops.baddbmm,
        Arg(),
        Arg(),
        Arg(),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    pass_dict=pass_pattern,
    extra_check=is_baddbmm_replacable,
)
def baddbmm_replacement(match, add_1, mat_1, mat_2, *, beta, alpha):
    def repl(add_1, mat_1, mat_2, beta, alpha):
        counters["zentorch"]["zentorch_baddbmm"] += 1
        return zt_ops.zentorch_baddbmm(add_1, mat_1, mat_2, beta=beta, alpha=alpha)

    match.replace_by_example(repl, [add_1, mat_1, mat_2, beta, alpha])


# convolution replacement
def is_convolution_op_replaceable(match):
    from ._compile_backend import conv_config

    # Replace only if torch.grad is disabled as ZenDNN implements
    # Convolution for inference only.
    # Replace only if enable_zentorch_conv_flag is enabled
    if not torch.is_grad_enabled() and conv_config.enable_zentorch_conv_flag:
        if match.args[0].target == at_ops.clone.default:
            input = match.args[0].args[0].meta["val"]
        else:
            input = match.args[0].meta["val"]

        if match.args[1].target == at_ops.clone.default:
            weight = match.args[1].args[0].meta["val"]
        else:
            weight = match.args[1].meta["val"]

        if input.is_contiguous(
            memory_format=torch.channels_last
        ) and weight.is_contiguous(memory_format=torch.channels_last):
            return True
        return False
    return False


# when function args are more than 4, we will create a list to un-wrap.
conv_args = [Arg() for _ in range(9)]


@register_graph_pattern(
    CallFunction(at_ops.convolution, *conv_args),
    pass_dict=pass_pattern,
    extra_check=is_convolution_op_replaceable,
)
def convolution_replacement(match, *args):
    def repl(*args):
        counters["zentorch"]["zentorch_convolution"] += 1
        return zt_ops.zentorch_convolution(*args)

    match.replace_by_example(repl, [*args])


def replace_with_zentorch_ops(gm):
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="replace_with_zentorch_ops",
    )
    if config.pattern_matcher:
        if hasattr(at_ops, "_scaled_dot_product_flash_attention_for_cpu") and hasattr(
            zt_ops, "zentorch_sdpa"
        ):
            # sdpa replacement
            @register_graph_pattern(
                CallFunction(
                    at_ops._scaled_dot_product_flash_attention_for_cpu,
                    Arg(),
                    Arg(),
                    Arg(),
                    dropout_p=KeywordArg("dropout_p"),
                    is_causal=KeywordArg("is_causal"),
                    attn_mask=KeywordArg("attn_mask"),
                    scale=KeywordArg("scale"),
                ),
                pass_dict=pass_pattern,
            )
            def sdpa_replacement(
                match: Match,
                query,
                key,
                value,
                dropout_p,
                is_causal,
                *,
                attn_mask,
                scale
            ):
                def repl(query, key, value, dropout_p, is_causal, attn_mask, scale):
                    counters["zentorch"]["zentorch_sdpa"] += 1
                    return zt_ops.zentorch_sdpa(
                        query,
                        key,
                        value,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        attn_mask=attn_mask,
                        scale=scale,
                    )

                match.replace_by_example(
                    repl,
                    [query, key, value, dropout_p, is_causal, attn_mask, scale],
                )

            # dropout_p and is_causal are ambiguous arguments as they can be sent as
            # either positional or keyword args so we require two registrations
            @register_graph_pattern(
                CallFunction(
                    at_ops._scaled_dot_product_flash_attention_for_cpu,
                    Arg(),
                    Arg(),
                    Arg(),
                    KeywordArg("dropout_p"),
                    KeywordArg("is_causal"),
                    attn_mask=KeywordArg("attn_mask"),
                    scale=KeywordArg("scale"),
                ),
                pass_dict=pass_pattern,
            )
            def sdpa_replacement_2(
                match: Match,
                query,
                key,
                value,
                dropout_p,
                is_causal,
                *,
                attn_mask,
                scale
            ):
                def repl(query, key, value, dropout_p, is_causal, attn_mask, scale):
                    counters["zentorch"]["zentorch_sdpa"] += 1
                    return zt_ops.zentorch_sdpa(
                        query,
                        key,
                        value,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        attn_mask=attn_mask,
                        scale=scale,
                    )

                match.replace_by_example(
                    repl,
                    [query, key, value, dropout_p, is_causal, attn_mask, scale],
                )

        # first we check if ipex has been imported anywhere in the code,
        # then we register the corresponding graph patterns
        if "intel_extension_for_pytorch" in sys.modules:
            ipex_ops = torch.ops.torch_ipex
            # rope replacement
            rope_args = [Arg() for _ in range(7)]

            @register_graph_pattern(
                CallFunction(ipex_ops.rotary_position_embedding, *rope_args),
                pass_dict=pass_pattern,
            )
            def rope_replacement(match, *args):
                def repl(*args):
                    counters["zentorch"]["zentorch_rope"] += 1
                    return zt_ops.zentorch_rope(*args)

                match.replace_by_example(repl, [*args])

            # mmha replacement
            mmha_args = [Arg() for _ in range(12)]

            @register_graph_pattern(
                CallFunction(
                    ipex_ops.masked_multihead_self_attention,
                    *mmha_args,
                ),
                pass_dict=pass_pattern,
            )
            def mmha_replacement(match, *args):
                def repl(*args):
                    counters["zentorch"]["zentorch_mmha"] += 1
                    return zt_ops.zentorch_masked_multihead_self_attention(
                        *args,
                    )

                match.replace_by_example(repl, [*args])

            # TODO: Re-enable the deepseek rope replacements when the models
            # using these kernels are added to testing.
            # # RoPE deepseek is only present in IPEX 2.6 or above
            # if hasattr(ipex_ops, "rotary_position_embedding_deepseek"):
            #     rope_deepseek_args = [Arg() for _ in range(9)]

            #     @register_graph_pattern(
            #         CallFunction(
            #             ipex_ops.rotary_position_embedding_deepseek, *rope_deepseek_args
            #         ),
            #         pass_dict=pass_pattern,
            #     )
            #     def rope_deepseek_replacement(match, *args):
            #         def repl(*args):
            #             counters["zentorch"]["zentorch_rope_deepseek"] += 1
            #             return zt_ops.zentorch_rope_deepseek(*args)

            #         match.replace_by_example(repl, [*args])

            # # RoPE deepseek_v2 is only present in IPEX 2.7 or above
            # if hasattr(ipex_ops, "rotary_position_embedding_deepseek_v2"):
            #     rope_deepseek_v2_args = [Arg() for _ in range(8)]

            #     @register_graph_pattern(
            #         CallFunction(
            #             ipex_ops.rotary_position_embedding_deepseek_v2, *rope_deepseek_v2_args
            #         ),
            #         pass_dict=pass_pattern,
            #     )
            #     def rope_deepseek_v2_replacement(match, *args):
            #         def repl(*args):
            #             counters["zentorch"]["zentorch_rope_deepseek_v2"] += 1
            #             return zt_ops.zentorch_rope_deepseek_v2(*args)

            #         match.replace_by_example(repl, [*args])

        GraphTransformObserver(gm, "pass_pattern").apply_graph_pass(pass_pattern.apply)
    gm.graph.lint()
    gm.recompile()
    return gm


def replace_with_composite_zentorch_ops(fx_graph: torch.fx.GraphModule):
    # TODO
    # As every custom/composite operator would have its own characteristics.
    # As there is no one frame that fits all, we need to have a seperate
    # function or replacement strategy for each composite operator.
    # This graph pass takes care of only straight chains of operators, so
    # pattern matcher based approach is the best for this.
    # As and when we have more composite operators, we can add them in the
    # pattern matcher and replace them with the respective zentorch ops.

    # As of now we'll proceed with a graph pass for embedding bag operator.
    # Remove this comment when the above TODO is resolved and the code is
    # pushed to pattern matcher based approach.

    for node in fx_graph.graph.nodes:
        if node.target != at_ops._embedding_bag.default:
            continue

        users = list(node.users.keys())
        if len(users) != 1:
            logger.warning(
                "There are more than one users of aten embedding bag."
                "Removal of get-item node and replacement with zentorch op"
                "not possible."
            )
            continue

        user_node = users[0]
        if user_node.target != operator.getitem:
            logger.info(
                "The user of aten embedding bag is not a get-item node."
                "Cannot remove a non-get-item node from the graph and thus"
                "cannot replace aten embedding bag with zentorch op."
            )
            continue

        if not is_zendnn_embedding_bag_supported(node.args[0].meta["val"]):
            logger.info(
                "zentorch embedding-bag is not supported for the combination of "
                "dtype: %s and embedding dimension: %s.",
                node.args[0].meta["val"].dtype,
                node.args[0].meta["val"].shape[1],
            )
            continue

        user_node.replace_all_uses_with(node)
        node.target = zt_ops.zentorch_embedding_bag.default
        fx_graph.graph.erase_node(user_node)

    fx_graph.graph.lint()
    fx_graph.recompile()

    return fx_graph
