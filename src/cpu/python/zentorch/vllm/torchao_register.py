# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
from zentorch._logging import get_logger

logger = get_logger(__name__)


def _register_int4_opaque_tensor_config():
    """
    Register Int4WeightOnlyOpaqueTensorConfig with vLLM's torchao config loader.

    This adds torchao.prototype.int4_opaque_tensor to the ALLOWED_AO_MODULES set,
    enabling vLLM to deserialize models quantized with Int4WeightOnlyOpaqueTensorConfig.
    """

    from torchao.core.config import ALLOWED_AO_MODULES

    # Add the int4_opaque_tensor prototype module to allowed modules
    module_to_add = "torchao.prototype.int4_opaque_tensor"
    ALLOWED_AO_MODULES.add(module_to_add)


def _register_int4_slice_op():
    """
    Register slice operation for Int4OpaqueTensor from torchao.

    This enables narrow() calls (which internally use aten.slice) to work correctly
    with Int4OpaqueTensor by registering a custom handler with torchao's dispatch system.
    """
    try:
        from torchao.prototype.int4_opaque_tensor import (
            Int4OpaqueTensor,
        )

        aten = torch.ops.aten
        implements = Int4OpaqueTensor.implements

        @implements([aten.slice.Tensor])
        def _(func, types, args, kwargs):
            tensor = args[0]
            if not isinstance(tensor, Int4OpaqueTensor):
                # Not an Int4OpaqueTensor, use default behavior
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)

            # Extract arguments from aten.slice call
            # aten.slice.Tensor signature: (self, dim=0, start=0, end=9223372036854775807, step=1)
            # TODO: Evalluate and remove kwargs
            dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
            start = args[2] if len(args) > 2 else kwargs.get("start", 0)
            end = args[3] if len(args) > 3 else kwargs.get("end", None)
            step = args[4] if len(args) > 4 else kwargs.get("step", 1)

            # Handle edge cases
            if start is None:
                start = 0
            if end is None:
                end = tensor.shape[dim]
            if step != 1:
                raise NotImplementedError(
                    f"Int4OpaqueTensor.slice: step != 1 is not supported, got step={step}"
                )

            # Convert to our custom slicing function
            length = end - start

            return _slice_int4_opaque_tensor(tensor, dim, start, length)

        def _slice_int4_opaque_tensor(
            tensor: Int4OpaqueTensor, dim: int, start: int, length: int
        ) -> Int4OpaqueTensor:
            assert tensor.ndim == 2, f"Expected 2D tensor, got {tensor.ndim}D"
            assert dim in (
                0,
                1,
            ), f"Only slicing along dim 0 or 1 is supported, got dim={dim}"
            assert start >= 0, f"start must be non-negative, got {start}"
            assert length > 0, f"length must be positive, got {length}"
            assert start + length <= tensor.shape[dim], (
                f"start + length ({start + length}) exceeds tensor dimension {dim} "
                f"size ({tensor.shape[dim]})"
            )

            group_size = tensor.block_size[1]
            N, K = tensor.shape

            if dim == 0:
                # Slice along output dimension N
                # qdata shape: (N, K/2) -> slice along dim 0
                sliced_qdata = tensor.qdata[start : start + length, :]

                # scale_and_zero shape: (K/group_size, N, 2) -> slice along dim 1 (N dimension)
                sliced_scale_and_zero = tensor.scale_and_zero[
                    :, start : start + length, :
                ]

                # act_pre_scale shape: (N,) -> slice along dim 0
                sliced_act_pre_scale = None
                if tensor.act_pre_scale is not None:
                    sliced_act_pre_scale = tensor.act_pre_scale[start : start + length]

                new_shape = torch.Size([length, K])

            else:  # dim == 1
                # Slice along input dimension K
                # qdata shape: (N, K/2) -> slice along dim 1, accounting for packing
                # Packed indices: each int32 contains 2 int4 values
                # To slice from start to start+length-1, we need packed elements
                # from start//2 to (start+length-1)//2 (inclusive)
                start_packed = start // 2
                end_packed = (start + length - 1) // 2 + 1  # +1 for inclusive end
                length_packed = end_packed - start_packed
                sliced_qdata = tensor.qdata[
                    :, start_packed : start_packed + length_packed
                ]

                # scale_and_zero shape: (K/group_size, N, 2) -> slice along dim 0 (group dimension)
                # Calculate which groups are included
                start_scale = start // group_size
                end_scale = (
                    start + length + group_size - 1
                ) // group_size  # ceiling division
                length_scale = end_scale - start_scale
                sliced_scale_and_zero = tensor.scale_and_zero[
                    start_scale : start_scale + length_scale, :, :
                ]

                # act_pre_scale remains unchanged (it's per output channel)
                sliced_act_pre_scale = tensor.act_pre_scale

                new_shape = torch.Size([N, length])

            return Int4OpaqueTensor(
                qdata=sliced_qdata,
                scale_and_zero=sliced_scale_and_zero,
                block_size=tensor.block_size,
                shape=new_shape,
                act_pre_scale=sliced_act_pre_scale,
            )

        logger.info(
            "[zentorch] Successfully registered slice op for Int4OpaqueTensor"
        )

    except ImportError:
        logger.info(
            "[zentorch] Int4OpaqueTensor slice patch deferred (torchao not available)"
        )
    except Exception:
        logger.exception(
            "[zentorch] Failed to register slice op for Int4OpaqueTensor"
        )
