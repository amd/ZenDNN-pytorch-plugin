# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
from pathlib import Path
import os
import shutil
from hypothesis import given, settings, Verbosity, seed, strategies as st
import inspect
import copy
from dataclasses import dataclass
import unittest

sys.path.append(str(Path(__file__).parent.parent))

from zentorch_test_utils import (  # noqa: 402 # noqa: F401
    BaseZentorchTestCase,
    Test_Data,
    run_tests,
    zentorch,
    has_zentorch,
    counters,
    supported_dtypes,
    supported_dtypes_def,
    qlinear_dtypes,
    skip_test_pt_2_0,
    skip_test_pt_2_1,
    skip_test_pt_2_3,
    skip_test_pt_2_4,
    reset_dynamo,
    freeze_opt,
    freeze_def_opt,
    test_with_freeze_opt,
    woq_dtypes,
    include_last_offset_opt,
    scale_grad_opt,
    mode_opt,
    sparse_opt,
    input_dim_opt,
    q_weight_list_opt,
    bias_opt,
    woq_qzeros_opt,
    group_size_opt,
    group_size_def_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
    q_linear_dtype_opt,
    conv_stride,
    conv_stride_def,
    conv_padding,
    conv_padding_def,
    at_ops,
    zt_ops,
    qlinear_eltwise_map,
    seq_length_opt,
    batch_size_opt,
    torch,
    DataTypes,
    SEED,

    # common variables
    b_range,
    m_range,
    k_range,
    n_range,
    p_range,
    q_range,
    matrix_dim_1_range,
    matrix_dim_2_range,
    matrix_dim_3_range,
    matrix_dim_4_range,

    # conv vars
    conv_bs_range,
    conv_c_range,
    conv_h_range,
    conv_wd_range,
    conv_oc_range,
    conv_kh_range,
    conv_kw_range,
    conv_stride,
    conv_padding,
    conv_dilation2,

    # emb vars
    emb_r_range,
    emb_w_range,
    emb_d_range,
    emb_mlp_opt,

    # mm vars
    mm_input_scaler_range,

    # woq variables
    woq_m_range,
    woq_x_range,
    woq_y_range,
    woq_k_range,
    woq_dtypes,
    woq_qzeros_nonzero_dim_range,

    # add_xD variables
    mm_add_1D_m_range,
    mm_add_1D_k_range,
    mm_add_1D_n_range,
    mm_add_2D_m_range,
    mm_add_2D_k_range,
    mm_add_2D_n_range,
    mm_add_3D_m_range,
    mm_add_3D_k_range,
    mm_add_3D_n_range,
    mm_add_3D_p_range,
    mm_add_3D_q_range,
)


path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class Zentorch_TestCase(BaseZentorchTestCase):
    def setUp(self):
        super().setUp()
        os.makedirs(os.path.join(path, "data"), exist_ok=True)
        self.data = Test_Data()

    def tearDown(self):
        del self.data
        shutil.rmtree(os.path.join(path, "data"))

    def skip_if_does_not_support_arg_combination_for_qlinear(
        self, bias_opt_idx, input_dtype, output_dtype
    ):
        if (
            self.data.bias_for_qlinear[bias_opt_idx] is None
            and input_dtype in ("float32", "bfloat16")
            and output_dtype not in (input_dtype, "int8", "uint8")
        ):
            self.skipTest(
                "Skipping test, if bias is None and input is floating-point, then "
                "output dtype has to match either input dtype or be any of int8 "
                "or uint8"
            )

        if (
            self.data.bias_for_qlinear[bias_opt_idx] is not None
            and self.data.bias_for_qlinear[bias_opt_idx].dtype == torch.float32
            and output_dtype == "bfloat16"
        ):
            self.skipTest(
                "Skipping test, if bias is fp32, then output dtype cannot be bf16."
            )

        if (
            self.data.bias_for_qlinear[bias_opt_idx] is not None
            and self.data.bias_for_qlinear[bias_opt_idx].dtype == torch.bfloat16
            and output_dtype == "float32"
        ):
            self.skipTest(
                "Skipping test, if bias is bf16, then output dtype cannot be fp32."
            )

        if (
            self.data.bias_for_qlinear[bias_opt_idx] is not None
            and input_dtype in ("float32", "bfloat16")
            and self.data.bias_for_qlinear[bias_opt_idx].dtype
            != self.data.get_torch_type(input_dtype)
        ):
            self.skipTest(
                "Skipping test, if bias is not None and input is floating-point, then "
                "bias dtype has to match input dtype"
            )


# In symmetric quantization, ZenTorch uses int8 data type for zero points,
# while in asymmetric quantization, it employs uint8.
# This approach aligns with standard quantization practices
# where quantization tools follow the same data type conventions for zero point values.
def get_comp_zero_points(zero_points):
    if zero_points is None or zero_points.dtype == torch.int8:
        return None
    else:
        return zero_points.to(torch.int32)


@dataclass
class HypothesisConstants:
    # Define enums
    y_int8_min = -128
    y_int8_max = 127
    zero_point_max = 255
    packing_ratio = 8


class AddmmTestCase(Zentorch_TestCase):
    time_out = 10000
    max_example_per_test = 20

    def getData(self):
        return self.data

    def createDataAddmm(
        self,
        dtype,
        b,
        m,
        k,
        n,
        M,
        T1,
        x1,
        y1,
        M2,
        M3,
        x2,
        y2,
        x,
        y,
        x3d,
        y3d,
        input,
        input3d
    ):
        self.data.create_data_addmm(
            dtype=dtype,
            b=b,
            m=m,
            k=k,
            n=n,
            M=M,
            T1=T1,
            x1=x1,
            y1=y1,
            M2=M2,
            M3=M3,
            x2=x2,
            y2=y2,
            x=x,
            y=y,
            x3d=x3d,
            y3d=y3d,
            input=input,
            input3d=input3d
        )

    def createDataAddXD(self, dtype, mm_add_1D, mm_add_2D, mm_add_3D):
        self.data.create_data_mm_add_xD(dtype, mm_add_1D, mm_add_2D, mm_add_3D)

    @seed(seed=SEED)
    @staticmethod
    # The @st.composite decorator is used to define custom Hypothesis strategies
    # for generating complex test data structures.
    @st.composite
    def tensor_addmm_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        bRange=b_range,
        mRange=m_range,
        kRange=k_range,
        nRange=n_range,
        matrix_dim_1_Range=matrix_dim_1_range,
        matrix_dim_2_Range=matrix_dim_2_range,
        matrix_dim_3_Range=matrix_dim_3_range,
        matrix_dim_4_Range=matrix_dim_4_range,
    ):
        dtype = draw(st.sampled_from(dtype_list))
        freeze = draw(st.sampled_from(freeze_list))
        b = draw(st.integers(bRange.get_min(), bRange.get_max()))
        m = draw(st.integers(mRange.get_min(), mRange.get_max()))
        k = draw(st.integers(kRange.get_min(), kRange.get_max()))
        n = draw(st.integers(nRange.get_min(), nRange.get_max()))
        matrix_dim_1 = draw(
            st.integers(
                matrix_dim_1_Range.get_min(),
                matrix_dim_1_Range.get_max()
            )
        )
        matrix_dim_2 = draw(
            st.integers(
                matrix_dim_2_Range.get_min(),
                matrix_dim_2_Range.get_max()
            )
        )
        matrix_dim_3 = draw(
            st.integers(
                matrix_dim_3_Range.get_min(),
                matrix_dim_3_Range.get_max()
            )
        )
        matrix_dim_4 = draw(
            st.integers(
                matrix_dim_4_Range.get_min(),
                matrix_dim_4_Range.get_max()
            )
        )

        torch_type = DataTypes.get_torch_type(dtype)

        M = [
            torch.randn(matrix_dim_1, matrix_dim_3).type(torch_type),
            torch.randn(matrix_dim_3).type(torch_type),
        ]

        T1 = torch.randn(2, matrix_dim_3, matrix_dim_3).type(torch_type)

        x1 = [
            torch.randn(matrix_dim_1, matrix_dim_2).type(torch_type),
            torch.randn(matrix_dim_2, matrix_dim_1).transpose(0, 1).type(torch_type),
        ]

        y1 = [
            torch.randn(matrix_dim_2, matrix_dim_3).type(torch_type),
            torch.randn(matrix_dim_3, matrix_dim_2).transpose(1, 0).type(torch_type),
        ]

        M2 = torch.randn(matrix_dim_1, matrix_dim_3, matrix_dim_4).type(torch_type)

        M3 = torch.randn(matrix_dim_4).type(torch_type)

        x2 = [
            torch.randn(matrix_dim_1, matrix_dim_3, matrix_dim_2).type(torch_type),
            torch.randn(
                matrix_dim_1,
                matrix_dim_2,
                matrix_dim_3
            ).transpose(1, 2).type(torch_type),
            torch.randn(
                matrix_dim_3,
                matrix_dim_1,
                matrix_dim_2
            ).transpose(0, 1).type(torch_type),
        ]

        y2 = [
            torch.randn(matrix_dim_1, matrix_dim_2, matrix_dim_4).type(torch_type),
            torch.randn(
                matrix_dim_1,
                matrix_dim_4,
                matrix_dim_2
            ).transpose(1, 2).type(torch_type),
            torch.randn(
                matrix_dim_4,
                matrix_dim_2,
                matrix_dim_1
            ).transpose(0, 2).type(torch_type),
        ]

        x = torch.randn(m, k).type(torch_type)
        y = torch.randn(k, n).type(torch_type)
        x3d = torch.randn(b, m, k).type(torch_type)
        y3d = torch.randn(b, k, n).type(torch_type)
        input = torch.randn(m, n).type(torch_type)
        input3d = torch.randn(b, m, n).type(torch_type)

        return (
            dtype,
            freeze,
            b,
            m,
            k,
            n,
            M,
            T1,
            x1,
            y1,
            M2,
            M3,
            x2,
            y2,
            x,
            y,
            x3d,
            y3d,
            input,
            input3d,
        )

    @staticmethod
    def hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        bRange=b_range,
        mRange=m_range,
        kRange=k_range,
        nRange=n_range,
        matrix_dim_1_Range=matrix_dim_1_range,
        matrix_dim_2_Range=matrix_dim_2_range,
        matrix_dim_3_Range=matrix_dim_3_range,
        matrix_dim_4_Range=matrix_dim_4_range,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "dtype_list is empty"

        def hypothesis_params_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            # The @settings() decorator is used to configure Hypothesis test parameters,
            # such as the maximum number of examples, timeout, and verbosity level.
            @settings(
                deadline=AddmmTestCase.time_out,
                max_examples=AddmmTestCase.max_example_per_test,
                verbosity=Verbosity.normal,
            )
            # The @given() decorator is used to define test inputs for Hypothesis-based tests.
            # It generates test cases by drawing values from the provided strategies.
            # Decorator execution order:
            # 1. @st.composite (if present) defines custom strategies.
            # 2. @given() generates test inputs using these strategies.
            # 3. @settings() configures test parameters like timeout and max examples.
            @given(
                val=AddmmTestCase.tensor_addmm_strategy(
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    bRange=bRange,
                    mRange=mRange,
                    kRange=kRange,
                    nRange=nRange,
                    matrix_dim_1_Range=matrix_dim_1_Range,
                    matrix_dim_2_Range=matrix_dim_2_Range,
                    matrix_dim_3_Range=matrix_dim_3_Range,
                    matrix_dim_4_Range=matrix_dim_4_Range
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                (
                    dtype,
                    freeze,
                    b,
                    m,
                    k,
                    n,
                    M,
                    T1,
                    x1,
                    y1,
                    M2,
                    M3,
                    x2,
                    y2,
                    x,
                    y,
                    x3d,
                    y3d,
                    input,
                    input3d,
                ) = val

                if (
                    not hasattr(obj, 'getData') or not isinstance(
                        obj.getData(),
                        Test_Data
                    )
                ):
                    raise RuntimeError(
                        "hypothesis_params_addmm_itr called with invalid object"
                    )

                obj.createDataAddmm(
                    dtype=dtype,
                    b=b,
                    m=m,
                    k=k,
                    n=n,
                    M=M,
                    T1=T1,
                    x1=x1,
                    y1=y1,
                    M2=M2,
                    M3=M3,
                    x2=x2,
                    y2=y2,
                    x=x,
                    y=y,
                    x3d=x3d,
                    y3d=y3d,
                    input=input,
                    input3d=input3d,
                )

                test_args = {
                    'dtype': dtype,
                    'freeze_opt': freeze,
                }

                required_args = inspect.signature(function).parameters.keys()

                function(
                    obj,
                    *args,
                    **{k: v for k, v in test_args.items() if k in required_args},
                    **kwargs
                )
                return

            return wrapper

        return hypothesis_params_itr_impl

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_add_xD_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        mm_add_1D_m_Range=mm_add_1D_m_range,
        mm_add_1D_k_Range=mm_add_1D_k_range,
        mm_add_1D_n_Range=mm_add_1D_n_range,
        mm_add_2D_m_Range=mm_add_2D_m_range,
        mm_add_2D_k_Range=mm_add_2D_k_range,
        mm_add_2D_n_Range=mm_add_2D_n_range,
        mm_add_3D_m_Range=mm_add_3D_m_range,
        mm_add_3D_k_Range=mm_add_3D_k_range,
        mm_add_3D_n_Range=mm_add_3D_n_range,
        mm_add_3D_p_Range=mm_add_3D_p_range,
        mm_add_3D_q_Range=mm_add_3D_q_range,
    ):
        dtype = draw(st.sampled_from(dtype_list))
        mm_add_1D_m = draw(
            st.integers(
                mm_add_1D_m_Range.get_min(),
                mm_add_1D_m_Range.get_max()
            )
        )
        mm_add_1D_k = draw(
            st.integers(
                mm_add_1D_k_Range.get_min(),
                mm_add_1D_k_Range.get_max()
            )
        )
        mm_add_1D_n = draw(
            st.integers(
                mm_add_1D_n_Range.get_min(),
                mm_add_1D_n_Range.get_max()
            )
        )
        mm_add_2D_m = draw(
            st.integers(
                mm_add_2D_m_Range.get_min(),
                mm_add_2D_m_Range.get_max()
            )
        )
        mm_add_2D_k = draw(
            st.integers(
                mm_add_2D_k_Range.get_min(),
                mm_add_2D_k_Range.get_max()
            )
        )
        mm_add_2D_n = draw(
            st.integers(
                mm_add_2D_n_Range.get_min(),
                mm_add_2D_n_Range.get_max()
            )
        )
        mm_add_3D_m = draw(
            st.integers(
                mm_add_3D_m_Range.get_min(),
                mm_add_3D_m_Range.get_max()
            )
        )
        mm_add_3D_k = draw(
            st.integers(
                mm_add_3D_k_Range.get_min(),
                mm_add_3D_k_Range.get_max()
            )
        )
        mm_add_3D_n = draw(
            st.integers(
                mm_add_3D_n_Range.get_min(),
                mm_add_3D_n_Range.get_max()
            )
        )
        mm_add_3D_p = draw(
            st.integers(
                mm_add_3D_p_Range.get_min(),
                mm_add_3D_p_Range.get_max()
            )
        )
        mm_add_3D_q = draw(
            st.integers(
                mm_add_3D_q_Range.get_min(),
                mm_add_3D_q_Range.get_max()
            )
        )

        torch_type = DataTypes.get_torch_type(dtype)

        mm_add_1D = [
            torch.rand(mm_add_1D_m, mm_add_1D_k).type(torch_type),
            torch.rand(mm_add_1D_k, mm_add_1D_n).type(torch_type),
            torch.rand(mm_add_1D_n).type(torch_type),
        ]

        mm_add_2D = [
            torch.rand(mm_add_2D_m, mm_add_2D_k).type(torch_type),
            torch.rand(mm_add_2D_k, mm_add_2D_n).type(torch_type),
            torch.rand(mm_add_2D_m, mm_add_2D_n).type(torch_type),
        ]

        mm_add_3D = [
            torch.rand(mm_add_3D_m, mm_add_3D_k).type(torch_type),
            torch.rand(mm_add_3D_k, mm_add_3D_n).type(torch_type),
            torch.rand(mm_add_3D_p, mm_add_3D_q, mm_add_3D_n).type(torch_type),
        ]

        return (
            dtype,
            mm_add_1D,
            mm_add_2D,
            mm_add_3D,
        )

    @staticmethod
    def hypothesis_params_add_xD_itr(
        dtype_list=supported_dtypes_def,
        mm_add_1D_m_Range=mm_add_1D_m_range,
        mm_add_1D_k_Range=mm_add_1D_k_range,
        mm_add_1D_n_Range=mm_add_1D_n_range,
        mm_add_2D_m_Range=mm_add_2D_m_range,
        mm_add_2D_k_Range=mm_add_2D_k_range,
        mm_add_2D_n_Range=mm_add_2D_n_range,
        mm_add_3D_m_Range=mm_add_3D_m_range,
        mm_add_3D_k_Range=mm_add_3D_k_range,
        mm_add_3D_n_Range=mm_add_3D_n_range,
        mm_add_3D_p_Range=mm_add_3D_p_range,
        mm_add_3D_q_Range=mm_add_3D_q_range,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "one or more required input lists are empty"

        def hypothesis_params_add_xD_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=AddmmTestCase.time_out,
                max_examples=AddmmTestCase.max_example_per_test,
                verbosity=Verbosity.normal,
            )
            @given(
                val=AddmmTestCase.tensor_add_xD_strategy(
                    dtype_list=dtype_list,
                    mm_add_1D_m_Range=mm_add_1D_m_Range,
                    mm_add_1D_k_Range=mm_add_1D_k_Range,
                    mm_add_1D_n_Range=mm_add_1D_n_Range,
                    mm_add_2D_m_Range=mm_add_2D_m_Range,
                    mm_add_2D_k_Range=mm_add_2D_k_Range,
                    mm_add_2D_n_Range=mm_add_2D_n_Range,
                    mm_add_3D_m_Range=mm_add_3D_m_Range,
                    mm_add_3D_k_Range=mm_add_3D_k_Range,
                    mm_add_3D_n_Range=mm_add_3D_n_Range,
                    mm_add_3D_p_Range=mm_add_3D_p_Range,
                    mm_add_3D_q_Range=mm_add_3D_q_Range,
                )
            )
            def wrapper(obj, val, *args, **kwargs):

                if (
                    not hasattr(obj, 'getData') or not isinstance(
                        obj.getData(),
                        Test_Data
                    )
                ):
                    raise RuntimeError(
                        "hypothesis_params_add_xD_itr called with invalid object"
                    )

                (
                    dtype,
                    mm_add_1D,
                    mm_add_2D,
                    mm_add_3D,
                ) = val

                obj.createDataAddXD(
                    dtype=dtype,
                    mm_add_1D=mm_add_1D,
                    mm_add_2D=mm_add_2D,
                    mm_add_3D=mm_add_3D
                )

                test_args = {
                    'dtype': dtype,
                }

                required_args = inspect.signature(function).parameters.keys()

                function(
                    obj,
                    *args,
                    **{k: v for k, v in test_args.items() if k in required_args},
                    **kwargs
                )
                return

            return wrapper

        return hypothesis_params_add_xD_itr_impl


class ConvTestCase(Zentorch_TestCase):
    time_out = 10000
    max_example_per_test = 20

    def getData(self):
        return self.data

    def createData(
        self,
        dtype,
        conv_input,
        conv_weight,
        conv_bias,
        stride,
        padding,
        dilation,
        output_padding,
        conv_input3d,
        conv_weight3d,
        dilation2
    ):
        self.data.create_data_conv(
            dtype=dtype,
            conv_input=conv_input,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            conv_input3d=conv_input3d,
            conv_weight3d=conv_weight3d,
            dilation2=dilation2,
        )

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_conv_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        stride_list=conv_stride_def,
        padding_list=conv_padding_def,
        conv_bs_Range=conv_bs_range,
        conv_c_Range=conv_c_range,
        conv_h_Range=conv_h_range,
        conv_wd_Range=conv_wd_range,
        conv_oc_Range=conv_oc_range,
        conv_kh_Range=conv_kh_range,
        conv_kw_Range=conv_kw_range,
        conv_dilation2_list=conv_dilation2,
    ):
        dtype = draw(st.sampled_from(dtype_list))
        freeze = draw(st.sampled_from(freeze_list))
        stride = draw(st.sampled_from(stride_list))
        padding = draw(st.sampled_from(padding_list))
        conv_bs = draw(st.integers(conv_bs_Range.get_min(), conv_bs_Range.get_max()))
        conv_c = draw(st.integers(conv_c_Range.get_min(), conv_c_Range.get_max()))
        conv_h = draw(st.integers(conv_h_Range.get_min(), conv_h_Range.get_max()))
        conv_wd = draw(st.integers(conv_wd_Range.get_min(), conv_wd_Range.get_max()))
        conv_oc = draw(st.integers(conv_oc_Range.get_min(), conv_oc_Range.get_max()))
        conv_kh = draw(st.integers(conv_kh_Range.get_min(), conv_kh_Range.get_max()))
        conv_kw = draw(st.integers(conv_kw_Range.get_min(), conv_kw_Range.get_max()))
        conv_dilation2 = draw(st.sampled_from(conv_dilation2_list))

        torch_type = DataTypes.get_torch_type(dtype)

        conv_input = torch.randn(
            conv_bs,
            conv_c,
            conv_h,
            conv_wd
        ).type(torch_type).to(memory_format=torch.channels_last)
        conv_weight = torch.randn(
            conv_oc,
            conv_c,
            conv_kh,
            conv_kw
        ).type(torch_type).to(memory_format=torch.channels_last)
        conv_bias = torch.randn(conv_oc).type(torch_type)

        stride = stride
        padding = padding
        dilation = [1, 1]
        output_padding = [0, 0]

        conv_input3d = torch.randn(conv_bs, conv_c, conv_kh).type(torch_type)
        conv_weight3d = torch.randn(conv_oc, conv_c, conv_kh).type(torch_type)
        dilation2 = conv_dilation2

        return (
            dtype,
            freeze,
            conv_input,
            conv_weight,
            conv_bias,
            stride,
            padding,
            dilation,
            output_padding,
            conv_input3d,
            conv_weight3d,
            dilation2,
        )

    @staticmethod
    def hypothesis_params_conv_itr(
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        stride_list=conv_stride_def,
        padding_list=conv_padding_def,
        conv_bs_Range=conv_bs_range,
        conv_c_Range=conv_c_range,
        conv_h_Range=conv_h_range,
        conv_wd_Range=conv_wd_range,
        conv_oc_Range=conv_oc_range,
        conv_kh_Range=conv_kh_range,
        conv_kw_Range=conv_kw_range,
        conv_dilation2_list=conv_dilation2,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "dtype_list is empty"

        def hypothesis_params_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=ConvTestCase.time_out,
                max_examples=ConvTestCase.max_example_per_test,
                verbosity=Verbosity.normal,
            )
            @given(
                val=ConvTestCase.tensor_conv_strategy(
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    stride_list=stride_list,
                    padding_list=padding_list,
                    conv_bs_Range=conv_bs_Range,
                    conv_c_Range=conv_c_Range,
                    conv_h_Range=conv_h_Range,
                    conv_wd_Range=conv_wd_Range,
                    conv_oc_Range=conv_oc_Range,
                    conv_kh_Range=conv_kh_Range,
                    conv_kw_Range=conv_kw_Range,
                    conv_dilation2_list=conv_dilation2_list,
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                (
                    dtype,
                    freeze,
                    conv_input,
                    conv_weight,
                    conv_bias,
                    stride,
                    padding,
                    dilation,
                    output_padding,
                    conv_input3d,
                    conv_weight3d,
                    dilation2,
                ) = val

                if (
                    not hasattr(obj, 'getData') or not isinstance(
                        obj.getData(),
                        Test_Data
                    )
                ):
                    raise RuntimeError(
                        "hypothesis_params_conv_itr called with invalid object"
                    )

                obj.createData(
                    dtype=dtype,
                    conv_input=conv_input,
                    conv_weight=conv_weight,
                    conv_bias=conv_bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    output_padding=output_padding,
                    conv_input3d=conv_input3d,
                    conv_weight3d=conv_weight3d,
                    dilation2=dilation2,
                )

                test_args = {
                    'dtype': dtype,
                    'freeze_opt': freeze,
                    'stride': stride,
                    'padding': padding,
                }

                required_args = inspect.signature(function).parameters.keys()

                function(
                    obj,
                    *args,
                    **{k: v for k, v in test_args.items() if k in required_args},
                    **kwargs
                )
                return

            return wrapper

        return hypothesis_params_itr_impl


class EmbTestCase(Zentorch_TestCase):
    time_out = 10000
    max_example_per_test = 20

    def getData(self):
        return self.data

    def createData(
        self,
        dtype,
        R,
        W,
        k,
        embedding_matrix,
        emb_input,
        offsets,
        mlp_inputs
    ):
        self.data.create_data_emb(
            dtype=dtype,
            R=R,
            W=W,
            k=k,
            embedding_matrix=embedding_matrix,
            emb_input=emb_input,
            offsets=offsets,
            mlp_inputs=mlp_inputs
        )

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_emb_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        emb_rRange=emb_r_range,
        emb_wRange=emb_w_range,
        emb_dRange=emb_d_range,
        emb_mlp_list=emb_mlp_opt,
    ):
        dtype = draw(st.sampled_from(dtype_list))
        freeze = draw(st.sampled_from(freeze_list))
        emb_r = draw(st.integers(emb_rRange.get_min(), emb_rRange.get_max()))
        emb_w = draw(st.integers(emb_wRange.get_min(), emb_wRange.get_max()))
        emb_d = draw(st.integers(emb_dRange.get_min(), emb_dRange.get_max()))
        emb_mlp = draw(st.sampled_from(emb_mlp_list))

        torch_type = DataTypes.get_torch_type(dtype)

        R = emb_r
        W = emb_w
        k = emb_d
        embedding_matrix = torch.randn(R, k).type(torch_type)
        emb_input = torch.randint(0, R, (W,))
        offsets = torch.tensor([0, W])
        mlp_inputs = torch.randn(emb_mlp, k).type(torch_type)

        return (
            dtype,
            freeze,
            R,
            W,
            k,
            embedding_matrix,
            emb_input,
            offsets,
            mlp_inputs,
        )

    @staticmethod
    def hypothesis_params_emb_itr(
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        emb_rRange=emb_r_range,
        emb_wRange=emb_w_range,
        emb_dRange=emb_d_range,
        emb_mlp_list=emb_mlp_opt,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "dtype_list is empty"

        def hypothesis_params_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=EmbTestCase.time_out,
                max_examples=EmbTestCase.max_example_per_test,
                verbosity=Verbosity.normal,
            )
            @given(
                val=EmbTestCase.tensor_emb_strategy(
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    emb_rRange=emb_rRange,
                    emb_wRange=emb_wRange,
                    emb_dRange=emb_dRange,
                    emb_mlp_list=emb_mlp_list,
                )
            )
            def wrapper(
                obj,
                val,
                *args,
                **kwargs
            ):
                if (
                    not hasattr(obj, 'getData') or not isinstance(
                        obj.getData(),
                        Test_Data
                    )
                ):
                    raise RuntimeError(
                        "hypothesis_params_emb_itr called with invalid object"
                    )

                (
                    dtype,
                    freeze,
                    R,
                    W,
                    k,
                    embedding_matrix,
                    emb_input,
                    offsets,
                    mlp_inputs,
                ) = val

                obj.createData(
                    dtype=dtype,
                    R=R,
                    W=W,
                    k=k,
                    embedding_matrix=embedding_matrix,
                    emb_input=emb_input,
                    offsets=offsets,
                    mlp_inputs=mlp_inputs,
                )

                # Prepare the arguments to pass to the test function
                test_args = {
                    'dtype': dtype,
                    'freeze_opt': freeze,
                }

                # Get the required argument names for the test function
                required_args = inspect.signature(function).parameters.keys()

                # Call the test function with the appropriate arguments
                function(
                    obj,
                    *args,
                    **{k: v for k, v in test_args.items() if k in required_args},
                    **kwargs
                )
                return

            return wrapper

        return hypothesis_params_itr_impl


class MMTestCase(Zentorch_TestCase):
    time_out = 10000
    max_example_per_test = 20

    def getData(self):
        return self.data

    def createData(
        self,
        dtype,
        b,
        m,
        k,
        n,
        x,
        y,
        result,
        input,
        input1d,
        input_scalar,
        empty_bias,
        result_m,
        result_1,
        A,
        B,
        x3d,
        y3d,
        input3d
    ):
        self.data.create_data_mm(
            dtype=dtype,
            b=b,
            m=m,
            k=k,
            n=n,
            x=x,
            y=y,
            result=result,
            input=input,
            input1d=input1d,
            input_scalar=input_scalar,
            empty_bias=empty_bias,
            result_m=result_m,
            result_1=result_1,
            A=A,
            B=B,
            x3d=x3d,
            y3d=y3d,
            input3d=input3d,
        )

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_mm_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        bRange=b_range,
        mRange=m_range,
        kRange=k_range,
        nRange=n_range,
        mm_input_scaler_Range=mm_input_scaler_range,
    ):
        dtype = draw(st.sampled_from(dtype_list))
        freeze = draw(st.sampled_from(freeze_list))
        b = draw(st.integers(bRange.get_min(), bRange.get_max()))
        m = draw(st.integers(mRange.get_min(), mRange.get_max()))
        k = draw(st.integers(kRange.get_min(), kRange.get_max()))
        n = draw(st.integers(nRange.get_min(), nRange.get_max()))
        mm_input_scalar = draw(
            st.integers(
                mm_input_scaler_Range.get_min(),
                mm_input_scaler_Range.get_max()
            )
        )

        torch_type = DataTypes.get_torch_type(dtype)

        x = torch.randn(m, k).type(torch_type)
        y = torch.randn(k, n).type(torch_type)
        result = torch.zeros(m, n).type(torch_type)

        input = torch.randn(m, n).type(torch_type)
        input1d = torch.randn(n).type(torch_type)

        if torch_type in [torch.bfloat16, torch.float32]:
            input_scalar = torch.rand(()).type(torch_type)
        else:
            input_scalar = torch.randint(0, mm_input_scalar, ()).type(torch_type)

        empty_bias = torch.zeros(0).type(torch_type)
        result_m = torch.zeros(int(m)).type(torch_type)
        result_1 = torch.zeros(1).type(torch_type)

        A = torch.randn(m, 1).type(torch_type)
        B = torch.randn(1, m).type(torch_type)

        x3d = torch.randn(b, m, k).type(torch_type)
        y3d = torch.randn(b, k, n).type(torch_type)
        input3d = torch.randn(b, m, n).type(torch_type)

        return (
            dtype,
            b,
            m,
            k,
            n,
            freeze,
            x,
            y,
            result,
            input,
            input1d,
            input_scalar,
            empty_bias,
            result_m,
            result_1,
            A,
            B,
            x3d,
            y3d,
            input3d,
        )

    @staticmethod
    def hypothesis_params_mm_itr(
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        bRange=b_range,
        mRange=m_range,
        kRange=k_range,
        nRange=n_range,
        mm_input_scaler_Range=mm_input_scaler_range,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "dtype_list is empty"

        def hypothesis_params_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=MMTestCase.time_out,
                max_examples=MMTestCase.max_example_per_test,
                verbosity=Verbosity.normal,
            )
            @given(
                val=MMTestCase.tensor_mm_strategy(
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    bRange=bRange,
                    mRange=mRange,
                    kRange=kRange,
                    nRange=nRange,
                    mm_input_scaler_Range=mm_input_scaler_Range,
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                (
                    dtype,
                    freeze,
                    b,
                    m,
                    k,
                    n,
                    x,
                    y,
                    result,
                    input,
                    input1d,
                    input_scalar,
                    empty_bias,
                    result_m,
                    result_1,
                    A,
                    B,
                    x3d,
                    y3d,
                    input3d,
                ) = val

                if (
                    not hasattr(obj, 'getData') or not isinstance(
                        obj.getData(),
                        Test_Data
                    )
                ):
                    raise RuntimeError(
                        "hypothesis_params_mm_itr called with invalid object"
                    )

                obj.createData(
                    dtype=dtype,
                    b=b,
                    m=m,
                    k=k,
                    n=n,
                    x=x,
                    y=y,
                    result=result,
                    input=input,
                    input1d=input1d,
                    input_scalar=input_scalar,
                    empty_bias=empty_bias,
                    result_m=result_m,
                    result_1=result_1,
                    A=A,
                    B=B,
                    x3d=x3d,
                    y3d=y3d,
                    input3d=input3d,
                )

                test_args = {
                    'dtype': dtype,
                    'freeze_opt': freeze,
                }

                required_args = inspect.signature(function).parameters.keys()

                function(
                    obj,
                    *args,
                    **{k: v for k, v in test_args.items() if k in required_args},
                    **kwargs
                )
                return

            return wrapper

        return hypothesis_params_itr_impl


class WOQTestCase(Zentorch_TestCase):
    time_out = 10000
    max_example_per_test = 20

    def getData(self):
        return self.data

    def createData(
        self,
        dtype,
        woq_m,
        woq_x,
        woq_y,
        woq_k,
        b,
        m,
        n,
        packing_ratio,
        group_size_val,
        woq_input,
        woq_add,
        woq_mul,
        woq_qweight,
        woq_scales,
        woq_qzeros,
        woq_qzeros_nonzero,
        woq_bias, input3d, input1d
    ):
        self.data.create_data_woq(
            dtype=dtype,
            woq_m=woq_m,
            woq_x=woq_x,
            woq_y=woq_y,
            woq_k=woq_k,
            b=b,
            m=m,
            n=n,
            packing_ratio=packing_ratio,
            group_size_val=group_size_val,
            woq_input=woq_input,
            woq_add=woq_add,
            woq_mul=woq_mul,
            woq_qweight=woq_qweight,
            woq_scales=woq_scales,
            woq_qzeros=woq_qzeros,
            woq_qzeros_nonzero=woq_qzeros_nonzero,
            woq_bias=woq_bias,
            input3d=input3d,
            input1d=input1d,
        )

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_woq_strategy(
        draw,
        woq_dtypes_list,
        input_dim_opt_list,
        bias_opt_list,
        woq_qzeros_opt_list,
        group_size_opt_list=group_size_def_opt,
        scales_dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        woq_m_Range=woq_m_range,
        woq_x_Range=woq_x_range,
        woq_y_Range=woq_y_range,
        woq_k_Range=woq_k_range,
        bRange=b_range,
        mRange=m_range,
        nRange=n_range,
        woq_qzeros_nonzero_dim_Range=woq_qzeros_nonzero_dim_range,
    ):
        dtype = draw(st.sampled_from(woq_dtypes_list))
        woq_input_dim = draw(st.sampled_from(input_dim_opt_list))
        woq_bias_idx = draw(st.sampled_from(bias_opt_list))
        woq_qzeros_idx = draw(st.sampled_from(woq_qzeros_opt_list))
        scales_dtype = draw(st.sampled_from(scales_dtype_list))
        freeze = draw(st.sampled_from(freeze_list))
        group_size_val = draw(st.sampled_from(group_size_opt_list))
        woq_m = draw(st.integers(woq_m_Range.get_min(), woq_m_Range.get_max()))
        woq_x = draw(st.integers(woq_x_Range.get_min(), woq_x_Range.get_max()))
        woq_y = draw(st.integers(woq_y_Range.get_min(), woq_y_Range.get_max()))
        woq_k = draw(st.integers(woq_k_Range.get_min(), woq_k_Range.get_max()))
        b = draw(st.integers(bRange.get_min(), bRange.get_max()))
        m = draw(st.integers(mRange.get_min(), mRange.get_max()))
        n = draw(st.integers(nRange.get_min(), nRange.get_max()))
        woq_qzeros_nonzero_dim = draw(
            st.integers(
                woq_qzeros_nonzero_dim_Range.get_min(),
                woq_qzeros_nonzero_dim_Range.get_max()
            )
        )

        torch_type = DataTypes.get_torch_type(dtype)
        constants = HypothesisConstants()
        group_size = group_size_val
        packing_ratio = constants.packing_ratio
        if group_size == -1:
            woq_k = woq_k * packing_ratio
            group_size = woq_k
        else:
            woq_k = woq_k * packing_ratio * group_size

        woq_n = woq_k

        woq_input = {
            2: torch.randn(woq_m, woq_k).type(torch_type),
            3: torch.randn(woq_m, woq_y, woq_k).type(torch_type),
            4: torch.randn(woq_m, woq_x, woq_y, woq_k).type(
                torch_type
            ),
        }
        woq_add = {
            2: torch.randn(woq_m, woq_n).type(torch_type),
            3: torch.randn(woq_m, woq_y, woq_n).type(torch_type),
            4: torch.randn(woq_m, woq_x, woq_y, woq_n).type(
                torch_type
            ),
        }
        woq_mul = {
            2: torch.randn(woq_m, woq_n).type(torch_type),
            3: torch.randn(woq_m, woq_y, woq_n).type(torch_type),
            4: torch.randn(woq_m, woq_x, woq_y, woq_n).type(
                torch_type
            ),
        }
        woq_qweight = torch.randn(woq_k, woq_n // packing_ratio).type(
            torch.int32
        )
        woq_qweight = {
            "bfloat16": copy.deepcopy(woq_qweight),
            "float32": copy.deepcopy(woq_qweight),
        }
        woq_scales = torch.randn(woq_k // group_size, woq_n).type(
            torch.bfloat16
        )
        woq_scales = {
            "bfloat16": copy.deepcopy(woq_scales),
            "float32": copy.deepcopy(woq_scales.type(torch.float32)),
        }
        woq_qzeros = [
            None,
            torch.zeros(
                woq_k // group_size, woq_n // packing_ratio
            ).type(torch.int32),
        ]
        woq_qzeros_nonzero = torch.randint(
            1,
            woq_qzeros_nonzero_dim,
            (woq_k // group_size, woq_n // packing_ratio)
        ).type(torch.int32)
        woq_bias = [
            None,
            torch.randn(woq_n).type(torch_type),
        ]

        input3d = torch.randn(b, m, n).type(torch_type)
        input1d = torch.randn(n).type(torch_type)

        return (
            dtype,
            freeze,
            woq_m,
            woq_x,
            woq_y,
            woq_k,
            b,
            m,
            n,
            packing_ratio,
            scales_dtype,
            woq_input_dim,
            woq_bias_idx,
            woq_qzeros_idx,
            group_size_val,
            woq_input,
            woq_add,
            woq_mul,
            woq_qweight,
            woq_scales,
            woq_qzeros,
            woq_qzeros_nonzero,
            woq_bias,
            input3d,
            input1d
        )

    @staticmethod
    def hypothesis_params_woq_itr(
        woq_dtypes_list,
        input_dim_opt_list,
        bias_opt_list,
        woq_qzeros_opt_list,
        group_size_opt_list=group_size_def_opt,
        scales_dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        woq_m_Range=woq_m_range,
        woq_x_Range=woq_x_range,
        woq_y_Range=woq_y_range,
        woq_k_Range=woq_k_range,
        bRange=b_range,
        mRange=m_range,
        nRange=n_range,
        woq_qzeros_nonzero_dim_Range=woq_qzeros_nonzero_dim_range,
    ):
        skip_reason = None
        if not all([woq_dtypes_list, input_dim_opt_list, bias_opt_list, woq_qzeros_opt_list, scales_dtype_list]):
            skip_reason = "one or more required input lists are empty"

        def hypothesis_params_woq_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=WOQTestCase.time_out,
                max_examples=WOQTestCase.max_example_per_test,
                verbosity=Verbosity.normal,
            )
            @given(
                val=WOQTestCase.tensor_woq_strategy(
                    woq_dtypes_list,
                    input_dim_opt_list,
                    bias_opt_list,
                    woq_qzeros_opt_list,
                    group_size_opt_list=group_size_opt_list,
                    scales_dtype_list=scales_dtype_list,
                    freeze_list=freeze_list,
                    woq_m_Range=woq_m_range,
                    woq_x_Range=woq_x_range,
                    woq_y_Range=woq_y_range,
                    woq_k_Range=woq_k_range,
                    bRange=bRange,
                    mRange=mRange,
                    nRange=nRange,
                    woq_qzeros_nonzero_dim_Range=woq_qzeros_nonzero_dim_Range,
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                (
                    dtype,
                    freeze,
                    woq_m,
                    woq_x,
                    woq_y,
                    woq_k,
                    b,
                    m,
                    n,
                    packing_ratio,
                    scales_dtype,
                    woq_input_dim,
                    woq_bias_idx,
                    woq_qzeros_idx,
                    group_size_val,
                    woq_input,
                    woq_add,
                    woq_mul,
                    woq_qweight,
                    woq_scales,
                    woq_qzeros,
                    woq_qzeros_nonzero,
                    woq_bias,
                    input3d,
                    input1d
                ) = val

                if (
                    not hasattr(obj, 'getData') or not isinstance(
                        obj.getData(),
                        Test_Data
                    )
                ):
                    raise RuntimeError(
                        "hypothesis_params_woq_itr called with invalid object"
                    )

                obj.createData(
                    dtype=dtype,
                    woq_m=woq_m,
                    woq_x=woq_x,
                    woq_y=woq_y,
                    woq_k=woq_k,
                    b=b,
                    m=m,
                    n=n,
                    packing_ratio=packing_ratio,
                    group_size_val=group_size_val,
                    woq_input=woq_input,
                    woq_add=woq_add,
                    woq_mul=woq_mul,
                    woq_qweight=woq_qweight,
                    woq_scales=woq_scales,
                    woq_qzeros=woq_qzeros,
                    woq_qzeros_nonzero=woq_qzeros_nonzero,
                    woq_bias=woq_bias,
                    input3d=input3d,
                    input1d=input1d,
                )

                test_args = {
                    'dtype': dtype,
                    'scales_dtype': scales_dtype,
                    'woq_input_dim': woq_input_dim,
                    'woq_bias_idx': woq_bias_idx,
                    'woq_qzeros_idx': woq_qzeros_idx,
                    'group_size_val': group_size_val,
                    'freeze_opt': freeze,
                }

                required_args = inspect.signature(function).parameters.keys()

                function(
                    obj,
                    *args,
                    **{k: v for k, v in test_args.items() if k in required_args},
                    **kwargs
                )
                return

            return wrapper

        return hypothesis_params_woq_itr_impl


class QLinearTestCase(Zentorch_TestCase):
    time_out = 10000
    max_example_per_test = 20

    def getData(self):
        return self.data

    def createData(
        self,
        dtype,
        b,
        m,
        p,
        q,
        k,
        n,
        y_int8_square,
        bias_for_qlinear_square,
        y_scales_square,
        y_zero_points_square,
        x_for_qlinear,
        y_int8,
        binary_input,
        bias_for_qlinear,
        x_scales,
        x_zero_points,
        y_scales,
        y_zero_points,
        output_scales,
        output_zero_points,
        wrong_scales_per_channel,
        wrong_zero_points_per_channel,
        y,
        input1d,
        x1,
        y1,
        x3d,
        y3d,
        input3d
    ):
        self.data.create_data_qlinear(
            dtype,
            b=b,
            m=m,
            p=p,
            q=q,
            k=k,
            n=n,
            y_int8_square=y_int8_square,
            bias_for_qlinear_square=bias_for_qlinear_square,
            y_scales_square=y_scales_square,
            y_zero_points_square=y_zero_points_square,
            x_for_qlinear=x_for_qlinear,
            y_int8=y_int8,
            binary_input=binary_input,
            bias_for_qlinear=bias_for_qlinear,
            x_scales=x_scales,
            x_zero_points=x_zero_points,
            y_scales=y_scales,
            y_zero_points=y_zero_points,
            output_scales=output_scales,
            output_zero_points=output_zero_points,
            wrong_scales_per_channel=wrong_scales_per_channel,
            wrong_zero_points_per_channel=wrong_zero_points_per_channel,
            y=y,
            input1d=input1d,
            x1=x1,
            y1=y1,
            x3d=x3d,
            y3d=y3d,
            input3d=input3d
        )

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_qlinear_strategy(
        draw,
        input_dim_opt_list,
        q_weight_list_opt_list,
        bias_opt_list,
        q_granularity_opt_list,
        q_zero_points_dtype_opt_list,
        q_linear_dtype_opt_list,
        dtype_list=qlinear_dtypes,
        freeze_list=freeze_def_opt,
        bRange=b_range,
        mRange=m_range,
        pRange=p_range,
        qRange=q_range,
        kRange=k_range,
        nRange=n_range,
        matrix_dim_1_Range=matrix_dim_1_range,
        matrix_dim_2_Range=matrix_dim_2_range,
        matrix_dim_3_Range=matrix_dim_3_range,
    ):
        constants = HypothesisConstants()
        input_dim = draw(st.sampled_from(input_dim_opt_list))
        q_weight = draw(st.sampled_from(q_weight_list_opt_list))
        bias = draw(st.sampled_from(bias_opt_list))
        q_granularity = draw(st.sampled_from(q_granularity_opt_list))
        q_zero_points_dtype = draw(st.sampled_from(q_zero_points_dtype_opt_list))
        q_linear_dtype = draw(st.sampled_from(q_linear_dtype_opt_list))
        dtype = draw(st.sampled_from(dtype_list))
        freeze = draw(st.sampled_from(freeze_list))
        b = draw(st.integers(bRange.get_min(), bRange.get_max()))
        m = draw(st.integers(mRange.get_min(), mRange.get_max()))
        p = draw(st.integers(pRange.get_min(), pRange.get_max()))
        q = draw(st.integers(qRange.get_min(), qRange.get_max()))
        k = draw(st.integers(kRange.get_min(), kRange.get_max()))
        n = draw(st.integers(nRange.get_min(), nRange.get_max()))
        matrix_dim_1 = draw(
            st.integers(
                matrix_dim_1_Range.get_min(),
                matrix_dim_1_Range.get_max()
            )
        )
        matrix_dim_2 = draw(
            st.integers(
                matrix_dim_2_Range.get_min(),
                matrix_dim_2_Range.get_max()
            )
        )
        matrix_dim_3 = draw(
            st.integers(
                matrix_dim_3_Range.get_min(),
                matrix_dim_3_Range.get_max()
            )
        )

        torch_type = DataTypes.get_torch_type(dtype)

        y_int8_square = [
            torch.randint(
                constants.y_int8_min,
                constants.y_int8_max,
                (k, k),
                dtype=torch.int8
            )
        ]
        bias_for_qlinear_square = [
            None,
            torch.randn(k).type(torch_type)
        ]
        y_scales_square = {
            "per_tensor": torch.randn((1,)).type(torch.float32),
            "per_channel": torch.randn(k).type(torch.float32),
        }
        y_zero_points_square = {
            "per_tensor": torch.tensor(0).type(torch.int8),
            "per_channel": torch.zeros(k).type(torch.int8),
        }
        x_for_qlinear = {
            "float32": {
                2: torch.randn(m, k).type(torch.float32),
                3: torch.randn(m, p, k).type(torch.float32),
                4: torch.randn(m, p, q, k).type(torch.float32),
            },
            "bfloat16": {
                2: torch.randn(m, k).type(torch.bfloat16),
                3: torch.randn(m, p, k).type(torch.bfloat16),
                4: torch.randn(m, p, q, k).type(torch.bfloat16),
            },
            "int8": {
                2: torch.randint(
                    constants.y_int8_min,
                    constants.y_int8_max,
                    (m, k)
                ).type(torch.int8),
                3: torch.randint(
                    constants.y_int8_min,
                    constants.y_int8_max,
                    (m, p, k)
                ).type(torch.int8),
                4: torch.randint(
                    constants.y_int8_min,
                    constants.y_int8_max,
                    (m, p, q, k)
                ).type(torch.int8),
            },
            "uint8": {
                2: torch.randint(
                    0,
                    constants.zero_point_max,
                    (m, k)
                ).type(torch.uint8),
                3: torch.randint(0, constants.zero_point_max, (
                    m,
                    p,
                    k
                )).type(torch.uint8),
                4: torch.randint(0, constants.zero_point_max, (
                    m,
                    p,
                    q,
                    k
                )).type(torch.uint8),
            },
        }
        y_int8 = [
            torch.randint(
                constants.y_int8_min,
                constants.y_int8_max,
                (k, n)).type(torch.int8).t(),
            torch.randint(
                constants.y_int8_min,
                constants.y_int8_max,
                (n, k)).type(torch.int8),
        ]
        binary_input = {
            2: torch.randn(m, n),
            3: torch.randn(m, p, n),
            4: torch.randn(m, p, q, n),
        }
        bias_for_qlinear = [
            None,
            torch.randn(n).type(torch_type),
        ]
        x_scales = {
            "per_tensor": torch.randn((1,)).type(torch.float32),
        }
        x_zero_points = {
            "per_tensor": {
                "float32": {
                    "int8": torch.tensor(0).type(torch.int8),
                    "uint8": torch.randint(
                        0,
                        constants.zero_point_max,
                        (1,)
                    ).type(torch.uint8),
                },
                "bfloat16": {
                    "int8": torch.tensor(0).type(torch.int8),
                    "uint8": torch.randint(
                        0,
                        constants.zero_point_max,
                        (1,)
                    ).type(torch.uint8),
                },
                "int8": {
                    "int8": torch.zeros(1).type(torch.int8),
                    "uint8": torch.tensor(0).type(torch.int8),
                },
                "uint8": {
                    "int8": torch.randint(
                        0,
                        constants.zero_point_max,
                        (1,)
                    ).type(torch.uint8),
                    "uint8": torch.randint(
                        0,
                        constants.zero_point_max,
                        (1,)
                    ).type(torch.uint8),
                },
            },
        }
        y_scales = {
            "per_tensor": torch.randn((1,)).type(torch.float32),
            "per_channel": torch.randn(n).type(torch.float32),
        }
        y_zero_points = {
            "per_tensor": torch.tensor(0).type(torch.int8),
            "per_channel": torch.zeros(n).type(torch.int8),
        }
        output_scales = {
            "per_tensor": {
                "float32": {
                    "positive_scales": None,
                },
                "bfloat16": {
                    "positive_scales": None,
                },
                "uint8": {
                    "positive_scales": torch.rand((1,)).type(torch.float32),
                },
                "int8": {
                    "positive_scales": torch.rand((1,)).type(torch.float32),
                },
            }
        }
        output_zero_points = {
            "per_tensor": {
                "float32": None,
                "bfloat16": None,
                "uint8": torch.randint(
                    0,
                    constants.zero_point_max,
                    (1,)
                ).type(torch.uint8),
                "int8": torch.zeros(1).type(torch.int8),
            },
        }
        wrong_scales_per_channel = torch.randn(n + 1).type(torch.float32)
        wrong_zero_points_per_channel = torch.zeros(n + 1).type(torch.int8)
        y = torch.randn(k, n).type(torch_type)
        input1d = torch.randn(n).type(torch_type)
        x1 = [
            torch.randn(matrix_dim_1, matrix_dim_2).type(torch_type),
            torch.randn(
                matrix_dim_2,
                matrix_dim_1
            ).transpose(0, 1).type(torch_type),
        ]
        y1 = [
            torch.randn(matrix_dim_2, matrix_dim_3).type(torch_type),
            torch.randn(
                matrix_dim_3,
                matrix_dim_2
            ).transpose(1, 0).type(torch_type),
        ]
        x3d = torch.randn(b, m, k).type(torch_type)
        y3d = torch.randn(b, k, n).type(torch_type)
        input3d = torch.randn(b, m, n).type(torch_type)

        return (
            dtype,
            freeze,
            b,
            m,
            p,
            q,
            k,
            n,
            input_dim,
            q_weight,
            bias,
            q_granularity,
            q_zero_points_dtype,
            q_linear_dtype,
            y_int8_square,
            bias_for_qlinear_square,
            y_scales_square,
            y_zero_points_square,
            x_for_qlinear,
            y_int8,
            binary_input,
            bias_for_qlinear,
            x_scales,
            x_zero_points,
            y_scales,
            y_zero_points,
            output_scales,
            output_zero_points,
            wrong_scales_per_channel,
            wrong_zero_points_per_channel,
            y,
            input1d,
            x1,
            y1,
            x3d,
            y3d,
            input3d,
        )

    @staticmethod
    def hypothesis_params_qlinear_itr(
        input_dim_opt_list,
        q_weight_list_opt_list,
        bias_opt_list,
        q_granularity_opt_list,
        q_zero_points_dtype_opt_list,
        q_linear_dtype_opt_list,
        dtype_list=qlinear_dtypes,
        freeze_list=freeze_def_opt,
        bRange=b_range,
        mRange=m_range,
        pRange=p_range,
        qRange=q_range,
        kRange=k_range,
        nRange=n_range,
        matrix_dim_1_Range=matrix_dim_1_range,
        matrix_dim_2_Range=matrix_dim_2_range,
        matrix_dim_3_Range=matrix_dim_3_range,
    ):
        skip_reason = None
        if not all([input_dim_opt_list, q_weight_list_opt_list, bias_opt_list, q_granularity_opt_list,
                    q_zero_points_dtype_opt_list, q_linear_dtype_opt_list, dtype_list]):
            skip_reason = "one or more required input lists are empty"

        def hypothesis_params_qlinear_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=QLinearTestCase.time_out,
                max_examples=QLinearTestCase.max_example_per_test,
                verbosity=Verbosity.normal,
            )
            @given(
                val=QLinearTestCase.tensor_qlinear_strategy(
                    input_dim_opt_list,
                    q_weight_list_opt_list,
                    bias_opt_list,
                    q_granularity_opt_list,
                    q_zero_points_dtype_opt_list,
                    q_linear_dtype_opt_list,
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    bRange=bRange,
                    mRange=mRange,
                    pRange=pRange,
                    qRange=qRange,
                    kRange=kRange,
                    nRange=nRange,
                    matrix_dim_1_Range=matrix_dim_1_range,
                    matrix_dim_2_Range=matrix_dim_2_range,
                    matrix_dim_3_Range=matrix_dim_3_range,
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                (
                    dtype,
                    freeze,
                    b,
                    m,
                    p,
                    q,
                    k,
                    n,
                    input_dim,
                    q_weight,
                    bias,
                    q_granularity,
                    q_zero_points_dtype,
                    q_linear_dtype,
                    y_int8_square,
                    bias_for_qlinear_square,
                    y_scales_square,
                    y_zero_points_square,
                    x_for_qlinear,
                    y_int8,
                    binary_input,
                    bias_for_qlinear,
                    x_scales,
                    x_zero_points,
                    y_scales,
                    y_zero_points,
                    output_scales,
                    output_zero_points,
                    wrong_scales_per_channel,
                    wrong_zero_points_per_channel,
                    y,
                    input1d,
                    x1,
                    y1,
                    x3d,
                    y3d,
                    input3d,
                ) = val

                if (
                    not hasattr(obj, 'getData') or not isinstance(
                        obj.getData(),
                        Test_Data
                    )
                ):
                    raise RuntimeError(
                        "hypothesis_params_qlinear_itr called with invalid object"
                    )

                obj.createData(
                    dtype=dtype,
                    b=b,
                    m=m,
                    p=p,
                    q=q,
                    k=k,
                    n=n,
                    y_int8_square=y_int8_square,
                    bias_for_qlinear_square=bias_for_qlinear_square,
                    y_scales_square=y_scales_square,
                    y_zero_points_square=y_zero_points_square,
                    x_for_qlinear=x_for_qlinear,
                    y_int8=y_int8,
                    binary_input=binary_input,
                    bias_for_qlinear=bias_for_qlinear,
                    x_scales=x_scales,
                    x_zero_points=x_zero_points,
                    y_scales=y_scales,
                    y_zero_points=y_zero_points,
                    output_scales=output_scales,
                    output_zero_points=output_zero_points,
                    wrong_scales_per_channel=wrong_scales_per_channel,
                    wrong_zero_points_per_channel=wrong_zero_points_per_channel,
                    y=y,
                    input1d=input1d,
                    x1=x1,
                    y1=y1,
                    x3d=x3d,
                    y3d=y3d,
                    input3d=input3d,
                )

                test_args = {
                    'dtype': dtype,
                    'input_dim': input_dim,
                    'q_weight_idx': q_weight,
                    'bias_opt_idx': bias,
                    'q_granularity_val': q_granularity,
                    'q_zero_points_dtype': q_zero_points_dtype,
                    'input_dtype': q_linear_dtype,
                    'output_dtype': q_linear_dtype,
                    'freeze_opt': freeze,
                }

                required_args = inspect.signature(function).parameters.keys()

                function(
                    obj,
                    *args,
                    **{k: v for k, v in test_args.items() if k in required_args},
                    **kwargs
                )
                return

            return wrapper

        return hypothesis_params_qlinear_itr_impl
