# *******************************************************************************************************************************************************
# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.3.0%2Bcpu/intel_extension_for_pytorch/transformers/models/reference/modules/attentions.py
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

        if k_shape[1] == 1 and q_shape[1] == 1:
            key_layer = key_layer.view(k_shape[0], k_shape[1], k_shape[2] * k_shape[3])
            query_layer = query_layer.view(
                q_shape[0], q_shape[1], q_shape[2] * q_shape[3]
            )

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
        if k_shape[1] == 1 and q_shape[1] == 1:
            key_layer = key_layer.view(k_shape[0], k_shape[1], k_shape[2], k_shape[3])
            query_layer = query_layer.view(
                q_shape[0], q_shape[1], q_shape[2], q_shape[3]
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
