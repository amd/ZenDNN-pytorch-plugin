# *******************************************************************************************************************************************************
# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.3.0%2Bcpu/intel_extension_for_pytorch/transformers/models/reference/modules/attentions.py
# RotaryEmbedding_forward was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.4.0%2Bcpu/intel_extension_for_pytorch/transformers/models/reference/fusions/mha_fusion.py
# *******************************************************************************************************************************************************
import torch


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
    past_len = kv_cache[0].shape[-2] if kv_cache is not None else 0
    # Modified Rope op for ChatGLM3 to mitigate the
    # dimensionality prolem with compile flow.
    # Added a view op for the decode phase to squeeze the dimensions
    # before the Rope op.
    # Expanding them back with a view after Rope op.
    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        k_shape = key_layer.shape
        q_shape = query_layer.shape

        key_layer = self._IPEXROPE(
            key_layer,
            torch.tensor(past_len),
            k_shape[-2],
            k_shape[-1],
            1,
            64,
        )
        query_layer = self._IPEXROPE(
            query_layer,
            torch.tensor(past_len),
            q_shape[-2],
            q_shape[-1],
            1,
            64,
        )

    if attention_mask is None:
        attention_mask = torch.ones(
            query_layer.size(0),
            1,
            past_len + query_layer.size(1),
            past_len + key_layer.size(1),
            dtype=torch.bool,
        )
        attention_mask.tril_()
        attention_mask = ~attention_mask
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
        context_layer.shape[0], context_layer.shape[1], self.projection_size
    )
    # output = self.dense(context_layer)
    return output, present


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
