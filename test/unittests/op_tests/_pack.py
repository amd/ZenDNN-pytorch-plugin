# ******************************************************************************
#  * Copyright (c) 2025 Advanced Micro Devices, Inc.
#  * All rights reserved.
#  ******************************************************************************
from typing import Optional, Callable, TypeVar
import torch
from functools import wraps

T = TypeVar("T", bound="PackMethod")


def pack_check(
    func: Callable[[T, torch.Tensor, bool], torch.Tensor],
) -> Callable[[T, torch.Tensor, bool], torch.Tensor]:
    @wraps(func)
    def wrapper(self: T, to_pack: torch.Tensor, reorder: bool = True) -> torch.Tensor:
        if to_pack.ndim > 2:
            raise ValueError("Pack: Only support 1 and 2 dimensions tensor")
        if self.quant_algo == "awq":
            to_pack = to_pack.t().contiguous()
        return func(self, to_pack, reorder)

    return wrapper


def unpack_check(
    func: Callable[[T, torch.Tensor, bool], torch.Tensor],
) -> Callable[[T, torch.Tensor, bool], torch.Tensor]:
    @wraps(func)
    def wrapper(self: T, to_unpack: torch.Tensor, reorder: bool = True) -> torch.Tensor:
        if to_unpack.ndim > 2:
            raise ValueError("Unpack: Only support 1 and 2 dimensions tensor")
        unpacked = func(self, to_unpack, reorder)
        if self.quant_algo == "awq":
            unpacked = unpacked.t().contiguous()
        return unpacked

    return wrapper


# When using AWQ, a transpose operation is required; otherwise, no action is needed.
class PackMethod:
    def __init__(self, quant_algo: Optional[str], dtype: str) -> None:
        self.quant_algo = quant_algo
        self.dtype = dtype

    @pack_check
    def pack(self, to_pack: torch.Tensor, reorder: bool) -> torch.Tensor:
        return to_pack

    @unpack_check
    def unpack(self, to_unpack: torch.Tensor, reorder: bool) -> torch.Tensor:
        return to_unpack


class Pack_4_bits(PackMethod):
    def __init__(self, quant_algo: Optional[str], dtype: str) -> None:
        super().__init__(quant_algo, dtype)

    def pack(
        self, to_pack: torch.Tensor, reorder: bool = True, transpose: bool = True
    ) -> torch.Tensor:
        if to_pack.ndim > 2:
            raise ValueError("Pack: Only support 1 and 2 dimensions tensor")
        if transpose:
            to_pack = to_pack.t().contiguous()
        if reorder:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            order_map = [0, 1, 2, 3, 4, 5, 6, 7]
        pack_num = 8
        if to_pack.ndim == 2:
            packed = torch.zeros(
                to_pack.shape[0],
                to_pack.shape[1] // pack_num,
                dtype=torch.int32,
                device=to_pack.device,
            )
            new_c = to_pack.shape[1] // pack_num
            for c in range(new_c):
                for i in range(pack_num):
                    # Use -3 as an example, high_position is 11111111,
                    # cause bit_or generate errors, so we can't use int4 directly
                    packed_col = to_pack[:, c * pack_num + order_map[i]]
                    if self.dtype == "int4":
                        packed_col = packed_col & 0x0F
                    packed[:, c] = torch.bitwise_or(
                        packed[:, c], torch.bitwise_left_shift(packed_col, i * 4)
                    )
        else:
            packed = torch.zeros(
                to_pack.shape[0] // pack_num, dtype=torch.int32, device=to_pack.device
            )
            new_c = to_pack.shape[0] // pack_num
            for c in range(new_c):
                for i in range(pack_num):
                    # Use -3 as an example, high_position is 11111111,
                    # cause bit_or generate errors, so we can't use int4 directly
                    packed_col = to_pack[c * pack_num + order_map[i]]
                    if self.dtype == "int4":
                        packed_col = packed_col & 0x0F
                    packed[c] = torch.bitwise_or(
                        packed[c], torch.bitwise_left_shift(packed_col, i * 4)
                    )
        return packed

    def unpack(self, to_unpack: torch.Tensor, reorder: bool = True) -> torch.Tensor:
        if to_unpack.ndim > 2:
            raise ValueError("Unpack: Only support 1 and 2 dimensions tensor")
        shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], device=to_unpack.device)
        ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
        if to_unpack.ndim == 2:
            unpacked = (
                (to_unpack.unsqueeze(-1) >> shifts.view(1, 1, -1))
                .view(to_unpack.shape[0], -1)
                .to(torch.int8)
            )
            if reorder:
                order_tensor = torch.arange(
                    unpacked.shape[-1],
                    dtype=torch.int32,
                    device=unpacked.device,
                )
                order_tensor = order_tensor.view(-1, 8)
                order_tensor = order_tensor[:, ORDER].view(-1)
                unpacked = unpacked[:, order_tensor]
        elif to_unpack.ndim == 1:
            unpacked = (
                (to_unpack.unsqueeze(-1) >> shifts.view(1, -1)).view(-1).to(torch.int8)
            )
            if reorder:
                order_tensor = torch.arange(
                    unpacked.shape[-1],
                    dtype=torch.int32,
                    device=unpacked.device,
                )
                order_tensor = order_tensor.view(-1, 8)
                order_tensor = order_tensor[:, ORDER].view(-1)
                unpacked = unpacked[order_tensor]
        unpacked &= 0b1111
        # Use -3 as an example, we have to restore 00001101 to 11111101,
        # so we can check the fourth digit of the unzipped number,
        # and if the fourth digit == 1 it proves that the number is negative
        if self.dtype == "int4":
            mask = (unpacked & 0x08).bool()
            unpacked[mask] = unpacked[mask] | 0xF0
        unpacked = unpacked.t().contiguous()
        return unpacked


class Pack_8_bits(PackMethod):
    def __init__(self, quant_algo: Optional[str], dtype: str) -> None:
        super().__init__(quant_algo, dtype)

    @pack_check
    def pack(self, to_pack: torch.Tensor, reorder: bool) -> torch.Tensor:
        if self.dtype == "uint8":
            return to_pack.to(torch.uint8).contiguous()
        else:
            return to_pack.to(torch.int8).contiguous()

    @unpack_check
    def unpack(self, to_unpack: torch.Tensor, reorder: bool) -> torch.Tensor:
        return to_unpack.to(torch.int32).contiguous()


def create_pack_method(quant_algo: Optional[str], dtype: str) -> PackMethod:
    # awq need transpose
    if dtype == "int4" or dtype == "uint4":
        return Pack_4_bits(quant_algo, dtype)
    elif dtype == "int8" or dtype == "uint8":
        return Pack_8_bits(quant_algo, dtype)
    else:
        return PackMethod(quant_algo, dtype)
