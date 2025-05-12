# *******************************************************************************************************************************************************
# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
#
# _GLM2Attention_forward was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.3.0%2Bcpu/intel_extension_for_pytorch/transformers/models/reference/modules/attentions.py
#
# RotaryEmbedding_forward was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.4.0%2Bcpu/intel_extension_for_pytorch/transformers/models/reference/fusions/mha_fusion.py
#
# MistralModel_forward was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.4.0%2Bcpu/intel_extension_for_pytorch/transformers/models/reference/models.py
#
# MixtralDecoderLayer_forward was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.4.0%2Bcpu/intel_extension_for_pytorch/transformers/models/reference/modules/decoder.py
#
# zentorch_prepare_4d_causal_attention_mask was sourced from
# https://github.com/huggingface/transformers/blob/v4.43.2/src/transformers/modeling_attn_mask_utils.py
# *******************************************************************************************************************************************************
import torch
from typing import Optional, Tuple, List, Union
from torch.nn import functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
)
from .._logging import get_logger

# make a logger for this file
logger = get_logger(__name__)


@torch.library.impl("zenops::longrope", "cpu")
def longrope(
    inv_freq,
    max_seq_len_cached,
    max_position_embeddings,
    sin_cos,
    sin_cached,
    cos_cached,
    sin_cos_long,
    sin_cached_long,
    cos_cached_long,
    seq_len,
    rope_type,
):
    if seq_len > max_seq_len_cached:
        seq_len = torch.tensor(seq_len)
        if rope_type == 1:  # Phi3ForCausalLM
            return (
                max_position_embeddings,
                sin_cos_long,
                sin_cached_long,
                cos_cached_long,
            )
        elif rope_type == 2:  # Falcon
            t = torch.arange(seq_len, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            sin_cos = torch.cat(
                (freqs.sin().repeat(1, 2), freqs.cos().repeat(1, 2)), dim=-1
            )
            emb = torch.cat((freqs, freqs), dim=-1).float()
            cos_cached = emb.cos()[None, :, :]
            sin_cached = emb.sin()[None, :, :]
            return seq_len, sin_cos, sin_cached, cos_cached
        else:  # Default
            t = torch.arange(seq_len, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            sin_cos = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_cached = emb.cos()[None, None, :, :]
            sin_cached = emb.sin()[None, None, :, :]
            return (
                seq_len,
                sin_cos,
                sin_cached[:, :, :seq_len, ...],
                cos_cached[:, :, :seq_len, ...],
            )
    return max_seq_len_cached, sin_cos, sin_cached, cos_cached


torch.library.define(
    "zenops::longrope",
    "(Tensor inv_freq, int max_seq_len_cached, Tensor max_position_embeddings, "
    + " Tensor sin_cos, Tensor sin_cached, Tensor cos_cached, "
    + " Tensor? sin_cos_long, Tensor? sin_cached_long, Tensor? cos_cached_long,"
    + " Tensor seq_len, Tensor rope_type) -> (int, Tensor, Tensor, Tensor)",
)


@torch.library.register_fake("zenops::longrope")
def meta_longrope(
    inv_freq,
    max_seq_len_cached,
    max_position_embeddings,
    sin_cos,
    sin_cached,
    cos_cached,
    sin_cos_long,
    sin_cached_long,
    cos_cached_long,
    seq_len,
    rope_type,
):
    sin_cos_ouput = torch.empty(sin_cos.shape)
    sin_cached_ouput = torch.empty(sin_cached.shape)
    cos_cached_ouput = torch.empty(cos_cached.shape)
    return max_seq_len_cached, sin_cos_ouput, sin_cached_ouput, cos_cached_ouput


# The decorator torch.compiler.disable introduces a graph break wherever this
# function is called. This function contains a torch.where which is causing
# graph breaks and the subsequent code might encounter recompilations. Using
# this decorator, we are making sure that we are executing the entire block
# in eager mode and the graph before and after this function call can execute
# without any other graph breaks.
@torch.compiler.disable
def mixtral_where_for_loop(
    block_sparse_moe,
    expert_mask,
    final_hidden_states,
    hidden_states,
    routing_weights,
    distributed,
) -> torch.Tensor:

    def fuse_index_mul_index_add_default(
        curr_state, top_x, idx, routing_weights, final_hidden_states
    ):
        routing_w = routing_weights.index[top_x, idx].unsqueeze(-1)
        curr_state = curr_state * routing_w
        final_hidden_states.index_add_(
            0, top_x, curr_state.squeeze(0).to(curr_state.dtype())
        )
        return final_hidden_states

    fuse_index_mul_index_add_op = None

    if (hidden_states.dtype == torch.bfloat16) and hasattr(
        torch.ops.zentorch, "fuse_index_mul_index_add"
    ):
        fuse_index_mul_index_add_op = torch.ops.zentorch.fuse_index_mul_index_add
    else:
        fuse_index_mul_index_add_op = fuse_index_mul_index_add_default

    # Loop over all available experts in the model
    # and perform the computation on each expert
    for expert_idx in range(block_sparse_moe.num_experts):
        expert_layer = block_sparse_moe.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        curr_state = hidden_states[top_x].unsqueeze(0)

        if curr_state.shape[1] != 0:
            curr_state = curr_state.squeeze(0)
            linear1 = torch.ops.zentorch.zentorch_mm(
                curr_state, expert_layer.w3.weight.t()
            )
            linear2 = torch.ops.zentorch.zentorch_mm_silu_mul(
                curr_state, expert_layer.w1.weight.t(), linear1
            )
            curr_state = torch.ops.zentorch.zentorch_mm(
                linear2, expert_layer.w2.weight.t()
            )
            curr_state = curr_state.unsqueeze(0)

        final_hidden_states = fuse_index_mul_index_add_op(
            curr_state,
            top_x,
            idx,
            routing_weights,
            final_hidden_states,
        )

    return final_hidden_states


def zentorch_prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):

    if attention_mask is not None and len(attention_mask.shape) == 2:
        # This op is not registered in meta registrations of zentorch
        # because of the dynamicity introduced by the
        # past_key_values_length. This variable is changing for every
        # token which is causing many recompiles and thus the
        # performance is degrading. So, this op will cause graph breaks
        # as of now which is better than too many recompiles.
        # TODO
        # Address the dynamicity issue and register this op in meta
        # registrations
        attention_mask = attention_mask.to(inputs_embeds.dtype)
        attention_mask = torch.ops.zentorch.prepare_4d_causal_attention_mask(
            attention_mask,
            inputs_embeds,
            past_key_values_length,
            torch.tensor(torch.finfo(inputs_embeds.dtype).min).contiguous(),
            sliding_window,
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        key_value_length = input_shape[-1] + past_key_values_length
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: \
                    {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape,
            # invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:

        input_shape = (input_shape[0], input_shape[-1])

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = None
        if input_shape[-1] > 1 or sliding_window is not None:

            bsz, tgt_len = input_shape
            mask = torch.full(
                (tgt_len, tgt_len),
                torch.finfo(inputs_embeds.dtype).min,
                device=inputs_embeds.device,
            )
            mask_cond = torch.arange(mask.size(-1), device=inputs_embeds.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

            mask = mask.to(inputs_embeds.dtype)

            if past_key_values_length > 0:
                mask = torch.cat(
                    [
                        torch.zeros(
                            tgt_len,
                            past_key_values_length,
                            dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device,
                        ),
                        mask,
                    ],
                    dim=-1,
                )

            # add lower triangular sliding window mask if necessary
            if sliding_window is not None:
                diagonal = past_key_values_length - sliding_window - 1

                context_mask = torch.tril(
                    torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal
                )
                mask.masked_fill_(context_mask, torch.finfo(inputs_embeds.dtype).min)

            attention_mask = mask[None, None, :, :].expand(
                bsz, 1, tgt_len, tgt_len + past_key_values_length
            )

    return attention_mask


def ChatGLMModel_forward(
    self,
    input_ids,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    full_attention_mask: Optional[torch.BoolTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = False,
):
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    batch_size, seq_length = input_ids.shape

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    if self.pre_seq_len is not None:
        if past_key_values is None:
            past_key_values = self.get_prompt(
                batch_size=batch_size,
                device=input_ids.device,
                dtype=inputs_embeds.dtype,
            )
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask.new_ones((batch_size, self.pre_seq_len)),
                    attention_mask,
                ],
                dim=-1,
            )

    # Initializing attention_mask
    # With IPEX 2.7 version changes, attention_mask is not
    # initialized if past_len != 0 from the second iteration
    # which causes IpexScaleDotProduct to receive a None mask
    # and produce invalid outputs. As a result, this invalid outputs
    # lead to past_key_values received by _GLM2Attention_forward is None.
    if attention_mask is not None and full_attention_mask is None:
        full_attention_mask = self.get_masks(
            input_ids, past_key_values, padding_mask=attention_mask
        )

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    # if position_ids is not None:
    #     rotary_pos_emb = rotary_pos_emb[position_ids]
    # else:
    #     rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    # rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds,
        full_attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
    )

    return tuple(
        v
        for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
        if v is not None
    )


# _GLM2Attention_forward is inspired from IPEX 2.3.0 (commit id: d3c5244)
def _GLM2Attention_forward(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb,
    kv_cache=None,
    use_cache=True,
):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    # =====================
    # Query, Key, and Value
    # =====================
    # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
    mixed_x_layer = self.query_key_value(hidden_states)
    mixed_x_layer = mixed_x_layer.transpose(0, 1)
    past_len = kv_cache[0].shape[-2] if kv_cache is not None else 0
    # Modified Rope op for ChatGLM3 to mitigate the
    # dimensionality prolem with compile flow.
    # Added a view op for the decode phase to squeeze the dimensions
    # before the Rope op.
    # Expanding them back with a view after Rope op.
    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        query_layer, key_layer, value_layer = self._IPEXROPE(
            mixed_x_layer,
            torch.tensor(past_len),
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            1,
            64,
            num_concats=3,
        )
    else:
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1]
                + (
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(
                mixed_x_layer, 3
            )
    if attention_mask is None and past_len == 0:
        attention_mask = torch.ones(
            query_layer.size(0),
            1,
            past_len + query_layer.size(1),
            past_len + key_layer.size(1),
            dtype=torch.bool,
        )
        attention_mask.tril_()
        attention_mask = ~attention_mask
    if attention_mask is not None:
        attention_mask = torch.where(attention_mask, float("-inf"), attention_mask)
    (
        attn_output,
        attn_weights,
        present,
    ) = self._IPEXScaleDotProduct(
        query_layer,
        key_layer,
        value_layer,
        self.factor,
        kv_cache,
        None,
        attention_mask,
    )
    context_layer = attn_output.permute(2, 0, 1, 3).contiguous()
    output = context_layer.reshape(
        context_layer.shape[0], context_layer.shape[1], -1
    )
    # output = self.dense(context_layer)
    return output, present


def RotaryEmbedding_forward(self, seq_len=None):
    rope_type = 0
    if self.model_backbone == "Phi3ForCausalLM" and hasattr(self, "long_factor"):
        rope_type = 1
    elif self.model_backbone in ["FalconForCausalLM", "RWForCausalLM"]:
        rope_type = 2
    if seq_len is not None:
        max_seq_len_cached, self.sin_cos, self.sin_cached, self.cos_cached = (
            torch.ops.zenops.longrope(
                torch.tensor(self.inv_freq).contiguous(),
                self.max_seq_len_cached,
                torch.tensor(self.max_position_embeddings).contiguous(),
                self.sin_cos.contiguous(),
                self.sin_cached.contiguous(),
                self.cos_cached.contiguous(),
                self.sin_cos_long.contiguous() if rope_type == 1 else None,
                self.sin_cached_long.contiguous() if rope_type == 1 else None,
                self.cos_cached_long.contiguous() if rope_type == 1 else None,
                torch.tensor(seq_len).contiguous(),
                torch.tensor(rope_type).contiguous(),
            )
        )
        self.max_seq_len_cached = max_seq_len_cached
    return self.sin_cos, self.sin_cached, self.cos_cached


# MistralModel_forward is inspired from IPEX 2.4.0
def MistralModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids"
            " and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if (
        attention_mask is not None
        and hasattr(self.config, "_flash_attn_2_enabled")
        and self.config._flash_attn_2_enabled
        and past_key_values is not None
    ):
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched "
                " generation with padding_side='right'"
                " this may lead to unexpected behaviour for"
                " Flash Attention version of Mistral. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if hasattr(self, "_prepare_decoder_attention_mask"):
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = zentorch_prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning(
            "`use_cache=True` is incompatible with gradient checkpointing."
        )
        use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def MixtralDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if "padding_mask" in kwargs:
        logger.warning(
            "Passing `padding_mask` is deprecated and will be removed in v4.37."
            " Please make sure use `attention_mask` instead.`"
        )
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape
        `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and
        value projection states
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
            See `attentions` under
            returned tensors for more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers.
            They are useful for computing the router loss, and
            should not be returned during inference.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned
            and can be used to speed up decoding
            (see `past_key_values`).
    """

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.o_proj(hidden_states)
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    # hidden_states, router_logits = self.block_sparse_moe(hidden_states)

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.block_sparse_moe.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, self.block_sparse_moe.top_k, dim=-1
    )
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=self.block_sparse_moe.num_experts
    ).permute(2, 1, 0)

    final_hidden_states = mixtral_where_for_loop(
        self.block_sparse_moe,
        expert_mask,
        final_hidden_states,
        hidden_states,
        routing_weights,
        self.distributed,
    )

    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )
    hidden_states, router_logits = final_hidden_states, router_logits

    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if output_router_logits:
        outputs += (router_logits,)

    return outputs


def MixtralModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and "
            "decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    past_key_values_length = 0
    seq_length_with_past = seq_length
    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning(
            "`use_cache=True` is incompatible with gradient checkpointing",
        )
        use_cache = False

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # 4d mask is passed through the layers
    sliding_window = (
        self.config.max_position_embeddings
        if self.config.sliding_window is None
        else self.config.sliding_window
    )

    attention_mask = zentorch_prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
        sliding_window=sliding_window,
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_router_logits = () if output_router_logits else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                output_router_logits,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_router_logits:
            all_router_logits += (layer_outputs[-1],)
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_router_logits,
            ]
            if v is not None
        )
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_logits,
    )
