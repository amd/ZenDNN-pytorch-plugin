# *******************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# *******************************************************************
"""
Custom Quantizers for DLRMv2 PT2E Quantization

This module contains the EmbeddingBagUInt4Quantizer for UINT4
quantization of embedding tables in DLRMv2.
"""

import logging

import torch

from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    OperatorConfig,
    QuantizationConfig,
)
from torchao.quantization.pt2e.observer import (
    PerChannelMinMaxObserver,
)

logger = logging.getLogger(__name__)


class EmbeddingBagUInt4Quantizer(Quantizer):
    """Custom quantizer for Embedding/EmbeddingBag with UINT4.

    UINT4 quantization packs 2 uint4 values into a single int8:
    - Each weight is quantized to 4 bits (0-15 range)
    - Two consecutive uint4 values are packed: (high << 4) | low

    Supports:
    - torch.ops.aten.embedding.default
    - torch.ops.aten.embedding_bag.default
    - torch.ops.aten._embedding_bag.default
    """

    def __init__(self):
        super().__init__()
        self.quantization_config = (
            self._get_embedding_uint4_config()
        )

    def _get_embedding_uint4_config(self) -> OperatorConfig:
        """Get quantization config for UINT4 weights."""
        weight_quantization_spec = QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0,
            quant_max=15,  # UINT4 range
            qscheme=torch.per_channel_affine,
            ch_axis=0,
            observer_or_fake_quant_ctr=(
                PerChannelMinMaxObserver.with_args(
                    eps=2**-12,
                    quant_min=0,
                    quant_max=15,
                    dtype=torch.uint8,
                )
            ),
        )
        quantization_config = QuantizationConfig(
            input_activation=None,
            output_activation=None,
            weight=weight_quantization_spec,
            bias=None,
        )

        ops = [
            [torch.ops.aten.embedding.default],
            [torch.ops.aten.embedding_bag.default],
            [torch.ops.aten._embedding_bag.default],
        ]

        return OperatorConfig(
            config=quantization_config, operators=ops
        )

    def annotate(
        self, model: torch.fx.GraphModule,
    ) -> torch.fx.GraphModule:
        """Annotate Embedding/EmbeddingBag ops for UINT4."""
        embedding_config = self.quantization_config
        annotated_count = 0

        for node in model.graph.nodes:
            target_str = str(node.target)

            is_embedding_op = (
                node.op == "call_function"
                and ('embedding' in target_str.lower())
            )

            if is_embedding_op:
                if embedding_config.config.weight is None:
                    raise ValueError(
                        "Embedding config must have valid "
                        "weight quantization spec"
                    )

                node.meta["quantization_annotation"] = (
                    QuantizationAnnotation(
                        input_qspec_map={
                            node.args[0]: (
                                embedding_config.config.weight
                            ),
                        }
                    )
                )
                annotated_count += 1

        logger.info(
            "[EmbeddingBagUInt4Quantizer] Annotated %d "
            "embedding operations",
            annotated_count,
        )
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        """Validate the model."""
        pass

    @classmethod
    def get_supported_operators(cls) -> list[OperatorConfig]:
        """Return supported operator configurations."""
        instance = cls()
        return [instance.quantization_config]
