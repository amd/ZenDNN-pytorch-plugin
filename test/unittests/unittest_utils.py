# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
from pathlib import Path
import os
from hypothesis import given, settings, Verbosity, seed, strategies as st
import inspect
from dataclasses import dataclass
import random
import unittest
import pickle
from datetime import datetime

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from zentorch_test_utils import (  # noqa: 402 # noqa: F401
    BaseZentorchTestCase,
    Range,
    Test_Data,
    run_tests,
    zentorch,
    has_zentorch,
    counters,
    default_tolerance,
    supported_dtypes,
    update_supported_dtypes,
    supported_dtypes_def,
    qlinear_dtypes,
    Q_LINEAR_DTYPE_OPT_DEF,
    skip_test_pt_2_0,
    skip_test_pt_2_1,
    skip_test_pt_2_3,
    skip_test_pt_2_4,
    reset_dynamo,
    freeze_opt,
    freeze_def_opt,
    test_with_freeze_opt,
    test_with_freeze_opt_and_cpp_wrapper,
    cpp_wrapper_opt,
    cpp_wrapper_def_opt,
    mode_opt,
    MODE_OPT_DEF,
    include_last_offset_opt,
    INCLUDE_LAST_OFFSET_OPT_DEF,
    sparse_opt,
    SPARSE_OPT_DEF,
    scale_grad_opt,
    SCALE_GRAD_OPT_DEF,
    input_dim_opt,
    INPUT_DIM_OPT_DEF,
    batch_opt,
    in_features_opt,
    out_features_opt,
    woq_bias_opt,
    q_weight_list_opt,
    Q_WEIGHT_LIST_OPT_DEF,
    bias_opt,
    BIAS_OPT_DEF,
    woq_qzeros_opt,
    group_size_opt,
    group_size_def_opt,
    q_granularity_opt,
    Q_GRANULARITY_OPT_DEF,
    q_zero_points_dtype_opt,
    Q_ZERO_POINTS_DTYPE_OPT_DEF,
    q_linear_dtype_opt,
    woq_group_size_def,
    DataTypes,
    conv_stride,
    conv_stride_def,
    conv_padding,
    conv_padding_def,
    at_ops,
    zt_ops,
    qlinear_eltwise_map,
    QLINEAR_ELTWISE_OPT_DEF,
    seq_length_opt,
    SEQ_LENGTH_OPT_DEF,
    batch_size_opt,
    BATCH_SIZE_OPT_DEF,
    mask_type_opt,
    MASK_OPT_DEF,
    num_heads_opt,
    NUM_HEADS_OPT_DEF,
    head_dim_opt,
    HEAD_DIM_OPT_DEF,
    torch,
    DataTypes,
    Range,
    SEED,
    # common variables
    B_RANGE,
    M_RANGE,
    K_RANGE,
    N_RANGE,
    DYNAMIC_QLINEAR_K_OPT,
    P_RANGE,
    Q_RANGE,
    MATRIX_DIM_1_RANGE,
    MATRIX_DIM_2_RANGE,
    MATRIX_DIM_3_RANGE,
    MATRIX_DIM_4_RANGE,
    # conv vars
    CONV_BS_RANGE,
    CONV_C_RANGE,
    CONV_H_RANGE,
    CONV_WD_RANGE,
    CONV_OC_RANGE,
    CONV_KH_RANGE,
    CONV_KW_RANGE,
    conv_stride,
    conv_padding,
    CONV_DILATION2,
    # emb vars
    EMB_R_RANGE,
    EMB_W_RANGE,
    QUANT_EMB_W_RANGE,
    EMB_D_RANGE,
    QUANT_EMB_D_RANGE,
    QUANT_EMB_NUM_RANGE,
    EMB_MLP_OPT,
    EMB_NUM_OF_BAGS,
    # mm vars
    MM_INPUT_SCALER_RANGE,
    # woq variables
    woq_dtypes,
    WOQ_X_RANGE,
    WOQ_Y_RANGE,
    # woq int4 opaque tensor variables
    WOQ_INT4_BATCH_RANGE,
    WOQ_INT4_OUT_FEATURES_OPT,
    WOQ_INT4_GROUP_SIZE_OPT,
    WOQ_INT4_IN_FEATURES_MULT_OPT,
    WOQ_INT4_BIAS_OPT,
    # group_matmul variables
    GROUP_MATMUL_NUM_EXPERTS,
    GROUP_MATMUL_M_VALUES,
    GROUP_MATMUL_K_VALUES,
    GROUP_MATMUL_N_VALUES,
    GROUP_MATMUL_D_VALUES,
    GROUP_MATMUL_K_OUT_VALUES,
    GROUP_MATMUL_TOPK_VALUES,
    GROUP_MATMUL_NUM_TOKENS_VALUES,
    GROUP_MATMUL_INT8_K_VALUES,
    GROUP_MATMUL_INT8_GATED_K_VALUES,
    # add_xD variables
    MM_ADD_1D_M_RANGE,
    MM_ADD_1D_K_RANGE,
    MM_ADD_1D_N_RANGE,
    MM_ADD_2D_M_RANGE,
    MM_ADD_2D_K_RANGE,
    MM_ADD_2D_N_RANGE,
    MM_ADD_3D_M_RANGE,
    MM_ADD_3D_K_RANGE,
    MM_ADD_3D_N_RANGE,
    MM_ADD_3D_P_RANGE,
    MM_ADD_3D_Q_RANGE,
    # rms_norm vars
    RMS_BATCH_SIZE_RANGE,
    RMS_HIDDEN_SIZE_RANGE,
)

path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DUMP_ERRORS = os.getenv("ZENTORCH_UNITTEST_DUMP_ERROR_TENSORS", "0").lower() in (
    "1",
    "true",
    "yes",
)
USE_RANDOM_SEED = os.getenv("ZENTORCH_RANDOM_SEED", "0").lower() in ("1", "true", "yes")


def getRandomSeed():
    """Generate a random seed between 1 and 10000"""
    import time

    if USE_RANDOM_SEED:
        # Use combination of time and process ID, modulo 9999 and add 1 to get range [1,10000]
        return ((int(time.time() * 1000000) ^ (os.getpid() << 16)) % 9999) + 1
    else:
        return SEED  # Use fixed seed defined in zentorch_test_utils


class Zentorch_TestCase(BaseZentorchTestCase):
    _seen_error_test_hashes = set()
    dump_errors = DUMP_ERRORS

    def setUp(self):
        super().setUp()
        if self.dump_errors:
            os.makedirs("error_dumps", exist_ok=True)
        self.data = Test_Data()

    def tearDown(self):
        del self.data

    @classmethod
    def create_error_test_hash(cls, outstr):
        """Create a hash from the test parameters string"""
        import hashlib

        return hashlib.md5(outstr.encode()).hexdigest()[:8]

    @classmethod
    def is_error_test_seen(cls, test_hash):
        """Check if this test hash has been seen before"""
        return test_hash in cls._seen_error_test_hashes

    @classmethod
    def mark_error_test_seen(cls, test_hash):
        """Mark this test hash as seen"""
        cls._seen_error_test_hashes.add(test_hash)

    def dump_error_to_pickle(self, error, function_name, val, test_args):
        """Save error data to pickle file if dumping is enabled"""
        if not self.dump_errors:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_dump = {
            "error_message": error,
            "function_name": function_name,
            "tensor_values": val,
            "test_args": test_args,
            "timestamp": timestamp,
        }

        pickle_file = f"error_dumps/error_dump_{function_name}_{timestamp}.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(error_dump, f)
        return pickle_file

    @staticmethod
    def replay_from_pickle(pickle_file):
        """
        Decorator that replays a test using data from a pickle file

        Usage:
        @QLinearTestCase.replay_from_pickle("error_dumps/error_dump_20240709.pkl")
        def test_qlinear_accuracy(self, dtype, input_dim, ...):
            # Your test code
        """

        def replay_decorator(function):
            def wrapper(obj, *args, **kwargs):
                # Load error data
                with open(pickle_file, "rb") as f:
                    error_data = pickle.load(f)

                val = error_data["tensor_values"]
                test_args = error_data["test_args"]

                # Create test data using stored values
                obj.createDataFromVal(val)

                # Get required args
                required_args = inspect.signature(function).parameters.keys()
                filtered_args = {
                    k: v for k, v in test_args.items() if k in required_args
                }

                # Call the test function
                return function(obj, **filtered_args)

            return wrapper

        return replay_decorator

    @staticmethod
    def handleException(
        self, errorStr, hypStr, functionName, decName, pklReplayFunction, val, test_args
    ):
        # Handle test exceptions with special case for SkipTest
        if isinstance(errorStr, unittest.SkipTest) or "SkipTest" in str(errorStr):
            return

        # Create hash of the test parameters
        test_hash = self.create_error_test_hash(hypStr)
        if not self.is_error_test_seen(test_hash):
            # Print strategy input values on failure
            self.mark_error_test_seen(test_hash)

            print("\n============================================")
            print(f"Incoming error: {errorStr!r} from {functionName!r}")
            print("============================================\n")

            # Dump error data if enabled
            pickle_file = self.dump_error_to_pickle(
                errorStr, functionName, val, test_args
            )
            if pickle_file:
                print(f"\nError data saved to: {pickle_file}")

            outstr = ""
            outstr += "from zentorch_test_utils import Range\n"
            outstr += f"@{decName}("
            outstr += f"{hypStr})"
            # Print reproduction decorator
            print("\nTo reproduce the error use the hypothesis decorator below -\n")
            print(outstr)

            if pickle_file:
                print("\n================== OR ======================\n")
                print(f"@{pklReplayFunction}({pickle_file!r})")

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

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
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
            input1d,
        ) = val
        self.createDataAddmm(
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
            input1d=input1d,
        )

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
        input3d,
        input1d,
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
            input3d=input3d,
            input1d=input1d,
        )

    def createDataAddXD(self, dtype, mm_add_1D, mm_add_2D, mm_add_3D):
        self.data.create_data_mm_add_xD(dtype, mm_add_1D, mm_add_2D, mm_add_3D)

    @staticmethod
    def replay_addxD_from_pickle(pickle_file):
        def replay_decorator(function):
            def wrapper(obj, *args, **kwargs):
                # Load error data
                with open(pickle_file, "rb") as f:
                    error_data = pickle.load(f)

                val = error_data["tensor_values"]
                test_args = error_data["test_args"]

                (
                    hypStr,
                    tensor_seed,
                    dtype,
                    mm_add_1D,
                    mm_add_2D,
                    mm_add_3D,
                ) = val
                # Create test data using stored values
                obj.createDataAddXD(dtype, mm_add_1D, mm_add_2D, mm_add_3D)

                # Get required args
                required_args = inspect.signature(function).parameters.keys()
                filtered_args = {
                    k: v for k, v in test_args.items() if k in required_args
                }

                # Call the test function
                return function(obj, **filtered_args)

            return wrapper

        return replay_decorator

    @seed(seed=SEED)
    @staticmethod
    # The @st.composite decorator is used to define custom Hypothesis strategies
    # for generating complex test data structures.
    @st.composite
    def tensor_addmm_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        bRange=B_RANGE,
        mRange=M_RANGE,
        kRange=K_RANGE,
        nRange=N_RANGE,
        matrix_dim_1_Range=MATRIX_DIM_1_RANGE,
        matrix_dim_2_Range=MATRIX_DIM_2_RANGE,
        matrix_dim_3_Range=MATRIX_DIM_3_RANGE,
        matrix_dim_4_Range=MATRIX_DIM_4_RANGE,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)

        dtype = draw(st.sampled_from(dtype_list))
        hypStr += f"dtype_list=[{dtype!r}], "
        freeze = draw(st.sampled_from(freeze_list))
        hypStr += f"freeze_list=[{freeze}], "
        cpp_wrapper = draw(st.sampled_from(cpp_wrapper_opt_list))
        hypStr += f"cpp_wrapper_opt_list=[{cpp_wrapper}], "
        b = draw(st.integers(bRange.get_min(), bRange.get_max()))
        hypStr += f"bRange=Range({b},{b}), "
        m = draw(st.integers(mRange.get_min(), mRange.get_max()))
        hypStr += f"mRange=Range({m},{m}), "
        k = draw(st.integers(kRange.get_min(), kRange.get_max()))
        hypStr += f"kRange=Range({k},{k}), "
        n = draw(st.integers(nRange.get_min(), nRange.get_max()))
        hypStr += f"nRange=Range({n},{n}), "

        matrix_dim_1 = draw(
            st.integers(matrix_dim_1_Range.get_min(), matrix_dim_1_Range.get_max())
        )
        hypStr += f"matrix_dim_1_Range=Range({matrix_dim_1},{matrix_dim_1}), "
        matrix_dim_2 = draw(
            st.integers(matrix_dim_2_Range.get_min(), matrix_dim_2_Range.get_max())
        )
        hypStr += f"matrix_dim_2_Range=Range({matrix_dim_2},{matrix_dim_2}), "
        matrix_dim_3 = draw(
            st.integers(matrix_dim_3_Range.get_min(), matrix_dim_3_Range.get_max())
        )
        hypStr += f"matrix_dim_3_Range=Range({matrix_dim_3},{matrix_dim_3}), "
        matrix_dim_4 = draw(
            st.integers(matrix_dim_4_Range.get_min(), matrix_dim_4_Range.get_max())
        )
        hypStr += f"matrix_dim_4_Range=Range({matrix_dim_4},{matrix_dim_4}), "

        torch_type = DataTypes.get_torch_type(dtype)

        M = [
            torch.randn(matrix_dim_1, matrix_dim_3, generator=generator).type(
                torch_type
            ),
            torch.randn(matrix_dim_3, generator=generator).type(torch_type),
        ]

        T1 = [
            torch.randn(2, matrix_dim_3, matrix_dim_3, generator=generator).type(
                torch_type
            )
        ]

        x1 = [
            torch.randn(matrix_dim_1, matrix_dim_2, generator=generator).type(
                torch_type
            ),
            torch.randn(matrix_dim_2, matrix_dim_1, generator=generator)
            .transpose(0, 1)
            .type(torch_type),
        ]

        y1 = [
            torch.randn(matrix_dim_2, matrix_dim_3, generator=generator).type(
                torch_type
            ),
            torch.randn(matrix_dim_3, matrix_dim_2, generator=generator)
            .transpose(1, 0)
            .type(torch_type),
        ]

        M2 = torch.randn(
            matrix_dim_1, matrix_dim_3, matrix_dim_4, generator=generator
        ).type(torch_type)

        M3 = torch.randn(matrix_dim_4, generator=generator).type(torch_type)

        x2 = [
            torch.randn(
                matrix_dim_1, matrix_dim_3, matrix_dim_2, generator=generator
            ).type(torch_type),
            torch.randn(matrix_dim_1, matrix_dim_2, matrix_dim_3, generator=generator)
            .transpose(1, 2)
            .type(torch_type),
            torch.randn(matrix_dim_3, matrix_dim_1, matrix_dim_2, generator=generator)
            .transpose(0, 1)
            .type(torch_type),
        ]

        y2 = [
            torch.randn(
                matrix_dim_1, matrix_dim_2, matrix_dim_4, generator=generator
            ).type(torch_type),
            torch.randn(matrix_dim_1, matrix_dim_4, matrix_dim_2, generator=generator)
            .transpose(1, 2)
            .type(torch_type),
            torch.randn(matrix_dim_4, matrix_dim_2, matrix_dim_1, generator=generator)
            .transpose(0, 2)
            .type(torch_type),
        ]

        x = torch.randn(m, k, generator=generator).type(torch_type)
        y = torch.randn(k, n, generator=generator).type(torch_type)
        x3d = torch.randn(b, m, k, generator=generator).type(torch_type)
        y3d = torch.randn(b, k, n, generator=generator).type(torch_type)
        input = torch.randn(m, n, generator=generator).type(torch_type)
        input1d = torch.randn(n, generator=generator).type(torch_type)
        input3d = torch.randn(b, m, n, generator=generator).type(torch_type)

        return (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
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
            input1d,
        )

    @staticmethod
    def hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        bRange=B_RANGE,
        mRange=M_RANGE,
        kRange=K_RANGE,
        nRange=N_RANGE,
        matrix_dim_1_Range=MATRIX_DIM_1_RANGE,
        matrix_dim_2_Range=MATRIX_DIM_2_RANGE,
        matrix_dim_3_Range=MATRIX_DIM_3_RANGE,
        matrix_dim_4_Range=MATRIX_DIM_4_RANGE,
        time_out=None,
        tensor_seed=0,
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
                deadline=AddmmTestCase.time_out if time_out is None else time_out,
                max_examples=AddmmTestCase.max_example_per_test,
                verbosity=Verbosity.quiet,
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
                    cpp_wrapper_opt_list=cpp_wrapper_opt_list,
                    bRange=bRange,
                    mRange=mRange,
                    kRange=kRange,
                    nRange=nRange,
                    matrix_dim_1_Range=matrix_dim_1_Range,
                    matrix_dim_2_Range=matrix_dim_2_Range,
                    matrix_dim_3_Range=matrix_dim_3_Range,
                    matrix_dim_4_Range=matrix_dim_4_Range,
                    tensor_seed=tensor_seed,
                ),
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    hypStr, tensor_seed, dtype, freeze, cpp_wrapper, *_ = val

                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_addmm_itr called with invalid object"
                        )

                    obj.createDataFromVal(val)

                    test_args = {
                        "dtype": dtype,
                        "freeze_opt": freeze,
                        "cpp_wrapper": cpp_wrapper,
                    }

                    required_args = inspect.signature(function).parameters.keys()

                    function(
                        obj,
                        *args,
                        **{k: v for k, v in test_args.items() if k in required_args},
                        **kwargs,
                    )
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "AddmmTestCase.hypothesis_params_addmm_itr"
                        pklReplayFunction = "AddmmTestCase.replay_from_pickle"
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise  # Re-raise the exception after printing
                return

            return wrapper

        return hypothesis_params_itr_impl

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_add_xD_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        mm_add_1D_m_Range=MM_ADD_1D_M_RANGE,
        mm_add_1D_k_Range=MM_ADD_1D_K_RANGE,
        mm_add_1D_n_Range=MM_ADD_1D_N_RANGE,
        mm_add_2D_m_Range=MM_ADD_2D_M_RANGE,
        mm_add_2D_k_Range=MM_ADD_2D_K_RANGE,
        mm_add_2D_n_Range=MM_ADD_2D_N_RANGE,
        mm_add_3D_m_Range=MM_ADD_3D_M_RANGE,
        mm_add_3D_k_Range=MM_ADD_3D_K_RANGE,
        mm_add_3D_n_Range=MM_ADD_3D_N_RANGE,
        mm_add_3D_p_Range=MM_ADD_3D_P_RANGE,
        mm_add_3D_q_Range=MM_ADD_3D_Q_RANGE,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)
        dtype = draw(st.sampled_from(dtype_list))
        hypStr += f"dtype_list=[{dtype!r}]"
        mm_add_1D_m = draw(
            st.integers(mm_add_1D_m_Range.get_min(), mm_add_1D_m_Range.get_max())
        )
        hypStr += f"mm_add_1D_m_Range=Range({mm_add_1D_m}, {mm_add_1D_m}), "
        mm_add_1D_k = draw(
            st.integers(mm_add_1D_k_Range.get_min(), mm_add_1D_k_Range.get_max())
        )
        hypStr += f"mm_add_1D_k_Range=Range({mm_add_1D_k}, {mm_add_1D_k}), "
        mm_add_1D_n = draw(
            st.integers(mm_add_1D_n_Range.get_min(), mm_add_1D_n_Range.get_max())
        )
        hypStr += f"mm_add_1D_n_Range=Range({mm_add_1D_n}, {mm_add_1D_n}), "
        mm_add_2D_m = draw(
            st.integers(mm_add_2D_m_Range.get_min(), mm_add_2D_m_Range.get_max())
        )
        hypStr += f"mm_add_2D_m_Range=Range({mm_add_2D_m}, {mm_add_2D_m}), "
        mm_add_2D_k = draw(
            st.integers(mm_add_2D_k_Range.get_min(), mm_add_2D_k_Range.get_max())
        )
        hypStr += f"mm_add_2D_k_Range=Range({mm_add_2D_k}, {mm_add_2D_k}), "
        mm_add_2D_n = draw(
            st.integers(mm_add_2D_n_Range.get_min(), mm_add_2D_n_Range.get_max())
        )
        hypStr += f"mm_add_2D_n_Range=Range({mm_add_2D_n}, {mm_add_2D_n}), "
        mm_add_3D_m = draw(
            st.integers(mm_add_3D_m_Range.get_min(), mm_add_3D_m_Range.get_max())
        )
        hypStr += f"mm_add_3D_m_Range=Range({mm_add_3D_m}, {mm_add_3D_m}), "
        mm_add_3D_k = draw(
            st.integers(mm_add_3D_k_Range.get_min(), mm_add_3D_k_Range.get_max())
        )
        hypStr += f"mm_add_3D_k_Range=Range({mm_add_3D_k}, {mm_add_3D_k}), "
        mm_add_3D_n = draw(
            st.integers(mm_add_3D_n_Range.get_min(), mm_add_3D_n_Range.get_max())
        )
        hypStr += f"mm_add_3D_n_Range=Range({mm_add_3D_n}, {mm_add_3D_n}), "
        mm_add_3D_p = draw(
            st.integers(mm_add_3D_p_Range.get_min(), mm_add_3D_p_Range.get_max())
        )
        hypStr += f"mm_add_3D_p_Range=Range({mm_add_3D_p}, {mm_add_3D_p}), "
        mm_add_3D_q = draw(
            st.integers(mm_add_3D_q_Range.get_min(), mm_add_3D_q_Range.get_max())
        )
        hypStr += f"mm_add_3D_q_Range=Range({mm_add_3D_q}, {mm_add_3D_q}), "

        torch_type = DataTypes.get_torch_type(dtype)

        mm_add_1D = [
            torch.rand(mm_add_1D_m, mm_add_1D_k, generator=generator).type(torch_type),
            torch.rand(mm_add_1D_k, mm_add_1D_n, generator=generator).type(torch_type),
            torch.rand(mm_add_1D_n, generator=generator).type(torch_type),
        ]

        mm_add_2D = [
            torch.rand(mm_add_2D_m, mm_add_2D_k, generator=generator).type(torch_type),
            torch.rand(mm_add_2D_k, mm_add_2D_n, generator=generator).type(torch_type),
            torch.rand(mm_add_2D_m, mm_add_2D_n, generator=generator).type(torch_type),
        ]

        mm_add_3D = [
            torch.rand(mm_add_3D_m, mm_add_3D_k, generator=generator).type(torch_type),
            torch.rand(mm_add_3D_k, mm_add_3D_n, generator=generator).type(torch_type),
            torch.rand(mm_add_3D_p, mm_add_3D_q, mm_add_3D_n, generator=generator).type(
                torch_type
            ),
        ]

        return (
            hypStr,
            tensor_seed,
            dtype,
            mm_add_1D,
            mm_add_2D,
            mm_add_3D,
        )

    @staticmethod
    def hypothesis_params_add_xD_itr(
        dtype_list=supported_dtypes_def,
        mm_add_1D_m_Range=MM_ADD_1D_M_RANGE,
        mm_add_1D_k_Range=MM_ADD_1D_K_RANGE,
        mm_add_1D_n_Range=MM_ADD_1D_N_RANGE,
        mm_add_2D_m_Range=MM_ADD_2D_M_RANGE,
        mm_add_2D_k_Range=MM_ADD_2D_K_RANGE,
        mm_add_2D_n_Range=MM_ADD_2D_N_RANGE,
        mm_add_3D_m_Range=MM_ADD_3D_M_RANGE,
        mm_add_3D_k_Range=MM_ADD_3D_K_RANGE,
        mm_add_3D_n_Range=MM_ADD_3D_N_RANGE,
        mm_add_3D_p_Range=MM_ADD_3D_P_RANGE,
        mm_add_3D_q_Range=MM_ADD_3D_Q_RANGE,
        time_out=None,
        tensor_seed=0,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "one or more required input lists are empty"

        def hypothesis_params_add_xD_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=AddmmTestCase.time_out if time_out is None else time_out,
                max_examples=AddmmTestCase.max_example_per_test,
                verbosity=Verbosity.quiet,
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
                    tensor_seed=tensor_seed,
                )
            )
            def wrapper(obj, val, *args, **kwargs):

                try:
                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_add_xD_itr called with invalid object"
                        )

                    (
                        hypStr,
                        tensor_seed,
                        dtype,
                        mm_add_1D,
                        mm_add_2D,
                        mm_add_3D,
                    ) = val

                    obj.createDataAddXD(
                        dtype=dtype,
                        mm_add_1D=mm_add_1D,
                        mm_add_2D=mm_add_2D,
                        mm_add_3D=mm_add_3D,
                    )

                    test_args = {
                        "dtype": dtype,
                    }

                    required_args = inspect.signature(function).parameters.keys()

                    function(
                        obj,
                        *args,
                        **{k: v for k, v in test_args.items() if k in required_args},
                        **kwargs,
                    )
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "AddmmTestCase.hypothesis_params_add_xD_itr"
                        pklReplayFunction = "AddmmTestCase.replay_addxD_from_pickle"
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise  # Re-raise the exception after printing
                return

            return wrapper

        return hypothesis_params_add_xD_itr_impl


class GroupMatmulTestCase(Zentorch_TestCase):
    """Base class for group_matmul hypothesis-based tests.

    Provides a composite strategy and decorator for generating randomized
    dimension parameters (num_experts, M, K, N, D, K_out, topk, num_tokens).
    """
    # Each example builds per-expert w13/w2 weights, biases, scales and MoE
    # routing tensors, making it much heavier than a single matmul, so we run
    # fewer examples than the default 20.
    # NOTE: time_out=20000 with max_example_per_test=10 produced flaky
    # failures; tracked in ZENAI-3838. Until that is resolved, keep the
    # default 10000 deadline and a reduced example count.
    time_out = 10000
    max_example_per_test = 5

    def getData(self):
        return self.data

    def createData(
        self,
        dtype,
        num_experts,
        M,
        K,
        N,
        D,
        K_out,
        topk,
        num_tokens,
        inputs,
        w13_bias_none,
        w13_weights,
        w13_weights_int8,
        w13_scales,
        w13_int8_raw,
        w13_weights_gated,
        w13_bias_gated,
        w2_weights_gated,
        w2_bias_gated,
        w13_weights_int8_gated,
        w13_scales_gated,
        w2_weights_int8_gated,
        w2_scales_gated,
        w13_weights_int8_square,
        w13_scales_square,
        w2_weights_int8_square,
        w2_scales_square,
        hidden_states,
        topk_indices,
        topk_weights_routing,
    ):
        self.data.create_data_group_matmul(
            dtype=dtype,
            num_experts=num_experts,
            M=M,
            K=K,
            N=N,
            D=D,
            K_out=K_out,
            topk=topk,
            num_tokens=num_tokens,
            inputs=inputs,
            w13_bias_none=w13_bias_none,
            w13_weights=w13_weights,
            w13_weights_int8=w13_weights_int8,
            w13_scales=w13_scales,
            w13_int8_raw=w13_int8_raw,
            w13_weights_gated=w13_weights_gated,
            w13_bias_gated=w13_bias_gated,
            w2_weights_gated=w2_weights_gated,
            w2_bias_gated=w2_bias_gated,
            w13_weights_int8_gated=w13_weights_int8_gated,
            w13_scales_gated=w13_scales_gated,
            w2_weights_int8_gated=w2_weights_int8_gated,
            w2_scales_gated=w2_scales_gated,
            w13_weights_int8_square=w13_weights_int8_square,
            w13_scales_square=w13_scales_square,
            w2_weights_int8_square=w2_weights_int8_square,
            w2_scales_square=w2_scales_square,
            hidden_states=hidden_states,
            topk_indices=topk_indices,
            topk_weights_routing=topk_weights_routing,
        )

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            cpp_wrapper,
            num_experts,
            M,
            K,
            N,
            D,
            K_out,
            topk,
            num_tokens,
            inputs,
            w13_bias_none,
            w13_weights,
            w13_weights_int8,
            w13_scales,
            w13_int8_raw,
            w13_weights_gated,
            w13_bias_gated,
            w2_weights_gated,
            w2_bias_gated,
            w13_weights_int8_gated,
            w13_scales_gated,
            w2_weights_int8_gated,
            w2_scales_gated,
            w13_weights_int8_square,
            w13_scales_square,
            w2_weights_int8_square,
            w2_scales_square,
            hidden_states,
            topk_indices,
            topk_weights_routing,
        ) = val
        self.createData(
            dtype=dtype,
            num_experts=num_experts,
            M=M,
            K=K,
            N=N,
            D=D,
            K_out=K_out,
            topk=topk,
            num_tokens=num_tokens,
            inputs=inputs,
            w13_bias_none=w13_bias_none,
            w13_weights=w13_weights,
            w13_weights_int8=w13_weights_int8,
            w13_scales=w13_scales,
            w13_int8_raw=w13_int8_raw,
            w13_weights_gated=w13_weights_gated,
            w13_bias_gated=w13_bias_gated,
            w2_weights_gated=w2_weights_gated,
            w2_bias_gated=w2_bias_gated,
            w13_weights_int8_gated=w13_weights_int8_gated,
            w13_scales_gated=w13_scales_gated,
            w2_weights_int8_gated=w2_weights_int8_gated,
            w2_scales_gated=w2_scales_gated,
            w13_weights_int8_square=w13_weights_int8_square,
            w13_scales_square=w13_scales_square,
            w2_weights_int8_square=w2_weights_int8_square,
            w2_scales_square=w2_scales_square,
            hidden_states=hidden_states,
            topk_indices=topk_indices,
            topk_weights_routing=topk_weights_routing,
        )

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_group_matmul_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        num_experts_Range=GROUP_MATMUL_NUM_EXPERTS,
        m_Range=GROUP_MATMUL_M_VALUES,
        k_list=GROUP_MATMUL_K_VALUES,
        n_Range=GROUP_MATMUL_N_VALUES,
        d_list=GROUP_MATMUL_D_VALUES,
        k_out_list=GROUP_MATMUL_K_OUT_VALUES,
        topk_list=GROUP_MATMUL_TOPK_VALUES,
        num_tokens_list=GROUP_MATMUL_NUM_TOKENS_VALUES,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)

        dtype = draw(st.sampled_from(dtype_list))
        hypStr += f"dtype_list=[{dtype!r}], "
        cpp_wrapper = draw(st.sampled_from(cpp_wrapper_opt_list))
        hypStr += f"cpp_wrapper_opt_list=[{cpp_wrapper}], "
        num_experts = draw(
            st.integers(num_experts_Range.get_min(), num_experts_Range.get_max())
        )
        hypStr += f"num_experts_Range=Range({num_experts},{num_experts}), "
        M = draw(st.integers(m_Range.get_min(), m_Range.get_max()))
        hypStr += f"m_Range=Range({M},{M}), "
        K = draw(st.sampled_from(k_list))
        hypStr += f"k_list=[{K}], "
        N = draw(st.integers(n_Range.get_min(), n_Range.get_max()))
        hypStr += f"n_Range=Range({N},{N}), "
        D = draw(st.sampled_from(d_list))
        K_out = draw(st.sampled_from(k_out_list))
        topk = draw(st.sampled_from(topk_list))
        num_tokens = draw(st.sampled_from(num_tokens_list))

        hypStr += (
            f"d_list=[{D}], k_out_list=[{K_out}], "
            f"topk_list=[{topk}], num_tokens_list=[{num_tokens}]"
        )

        torch_type = DataTypes.get_torch_type(dtype)

        def _quantize_per_channel(fp32_list, out_features):
            """Per-channel symmetric int8 quantisation along axis 0."""
            int8_list, scale_list = [], []
            for w in fp32_list:
                s = w.abs().amax(dim=1).clamp(min=1e-12) / 127.0
                zp = torch.zeros(out_features, dtype=torch.long)
                wq = torch.quantize_per_channel(
                    w, s, zp, axis=0, dtype=torch.qint8
                )
                int8_list.append(wq.int_repr())
                scale_list.append(
                    wq.q_per_channel_scales().to(torch.float32)
                )
            return int8_list, scale_list

        # ---- Plain per-expert activations [M, K] ----
        inputs = [
            torch.randn(M, K, generator=generator).type(torch_type)
            for _ in range(num_experts)
        ]

        # ---- Per-expert "no bias" list ----
        w13_bias_none = [None] * num_experts

        w13_weights = [
            torch.randn(N, K, generator=generator).type(torch_type)
            for _ in range(num_experts)
        ]

        _w13_weights_primary_fp32 = [
            torch.randn(N, K, generator=generator) for _ in range(num_experts)
        ]
        w13_weights_int8, w13_scales = _quantize_per_channel(
            _w13_weights_primary_fp32, out_features=N
        )

        # ---- Raw int8 w13 [N, K] for the negative test ----
        w13_int8_raw = [
            torch.randint(
                -128, 128, (N, K), dtype=torch.int8, generator=generator
            )
            for _ in range(num_experts)
        ]

        # ---- Gated-shape weights / biases ----
        # N_act = 2*D so the kernel halves the gate / up split to D columns;
        # K_out is forced to K so W2 buffer-reuse stays valid.
        N_act = 2 * D
        w13_weights_gated = [
            torch.randn(N_act, K, generator=generator).type(torch_type)
            for _ in range(num_experts)
        ]
        w13_bias_gated = [
            torch.randn(N_act, generator=generator).type(torch_type)
            for _ in range(num_experts)
        ]
        w2_weights_gated = [
            torch.randn(K, D, generator=generator).type(torch_type)
            for _ in range(num_experts)
        ]
        w2_bias_gated = [
            torch.randn(K, generator=generator).type(torch_type)
            for _ in range(num_experts)
        ]

        # Int8 gated set: per-channel quantised w13 [N_act, K] and
        # w2 [K, D], plus matching fp32 references and bf16/f32 biases.
        w13_weights_gated_fp32 = [
            torch.randn(N_act, K, generator=generator)
            for _ in range(num_experts)
        ]
        w13_weights_int8_gated, w13_scales_gated = _quantize_per_channel(
            w13_weights_gated_fp32, out_features=N_act
        )

        w2_weights_gated_fp32 = [
            torch.randn(K, D, generator=generator)
            for _ in range(num_experts)
        ]
        w2_weights_int8_gated, w2_scales_gated = _quantize_per_channel(
            w2_weights_gated_fp32, out_features=K
        )

        # Used by sub-test 1 of test_int8_w13_and_w2_single_pass, which runs
        # cascaded W13 -> W2 with activation="none" and N=K_out=K so the
        # kernel can write W2 output back into the input buffers in place.
        w13_weights_int8_square_fp32 = [
            torch.randn(K, K, generator=generator)
            for _ in range(num_experts)
        ]
        w13_weights_int8_square, w13_scales_square = _quantize_per_channel(
            w13_weights_int8_square_fp32, out_features=K
        )
        w2_weights_int8_square_fp32 = [
            torch.randn(K, K, generator=generator)
            for _ in range(num_experts)
        ]
        w2_weights_int8_square, w2_scales_square = _quantize_per_channel(
            w2_weights_int8_square_fp32, out_features=K
        )

        # ---- MoE routing tensors keyed on (num_tokens, K, topk) ----
        hidden_states = torch.randn(num_tokens, K, generator=generator).type(
            torch_type
        )
        topk_indices = torch.randint(
            0, num_experts, (num_tokens, topk), generator=generator
        )
        topk_weights_routing = torch.rand(
            num_tokens, topk, generator=generator
        )

        return (
            hypStr,
            tensor_seed,
            dtype,
            cpp_wrapper,
            num_experts,
            M,
            K,
            N,
            D,
            K_out,
            topk,
            num_tokens,
            inputs,
            w13_bias_none,
            w13_weights,
            w13_weights_int8,
            w13_scales,
            w13_int8_raw,
            w13_weights_gated,
            w13_bias_gated,
            w2_weights_gated,
            w2_bias_gated,
            w13_weights_int8_gated,
            w13_scales_gated,
            w2_weights_int8_gated,
            w2_scales_gated,
            w13_weights_int8_square,
            w13_scales_square,
            w2_weights_int8_square,
            w2_scales_square,
            hidden_states,
            topk_indices,
            topk_weights_routing,
        )

    @staticmethod
    def hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes_def,
        num_experts_Range=GROUP_MATMUL_NUM_EXPERTS,
        m_Range=GROUP_MATMUL_M_VALUES,
        k_list=GROUP_MATMUL_K_VALUES,
        n_Range=GROUP_MATMUL_N_VALUES,
        d_list=GROUP_MATMUL_D_VALUES,
        k_out_list=GROUP_MATMUL_K_OUT_VALUES,
        topk_list=GROUP_MATMUL_TOPK_VALUES,
        num_tokens_list=GROUP_MATMUL_NUM_TOKENS_VALUES,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        time_out=None,
        tensor_seed=0,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "dtype_list is empty"

        def hypothesis_params_group_matmul_itr_impl(function):
            if skip_reason:
                print(
                    f"Skipping test - {function.__name__}: {skip_reason}"
                )
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=(
                    GroupMatmulTestCase.time_out
                    if time_out is None
                    else time_out
                ),
                max_examples=GroupMatmulTestCase.max_example_per_test,
                verbosity=Verbosity.quiet,
                database=None,
            )
            @given(
                val=GroupMatmulTestCase.tensor_group_matmul_strategy(
                    dtype_list=dtype_list,
                    cpp_wrapper_opt_list=cpp_wrapper_opt_list,
                    num_experts_Range=num_experts_Range,
                    m_Range=m_Range,
                    k_list=k_list,
                    n_Range=n_Range,
                    d_list=d_list,
                    k_out_list=k_out_list,
                    topk_list=topk_list,
                    num_tokens_list=num_tokens_list,
                    tensor_seed=tensor_seed,
                ),
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_group_matmul_itr called with "
                            "invalid object"
                        )

                    (
                        hypStr,
                        tensor_seed_val,
                        dtype,
                        cpp_wrapper,
                        num_experts,
                        M,
                        K,
                        N,
                        D,
                        K_out,
                        topk,
                        num_tokens,
                        *_tensor_payload,
                    ) = val

                    # Hypothesis example is fully reproducible from the
                    # `tensor_seed` printed in hypStr / the failure decorator.
                    torch.manual_seed(tensor_seed_val)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(tensor_seed_val)
                    np.random.seed(tensor_seed_val)
                    random.seed(tensor_seed_val)

                    obj.createDataFromVal(val)

                    test_args = {
                        "dtype": dtype,
                        "num_experts": num_experts,
                        "M": M,
                        "K": K,
                        "N": N,
                        "D": D,
                        "K_out": K_out,
                        "topk": topk,
                        "num_tokens": num_tokens,
                        "cpp_wrapper": cpp_wrapper,
                    }

                    required_args = (
                        inspect.signature(function).parameters.keys()
                    )

                    function(
                        obj,
                        *args,
                        **{
                            k: v
                            for k, v in test_args.items()
                            if k in required_args
                        },
                        **kwargs,
                    )
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = (
                            "GroupMatmulTestCase"
                            ".hypothesis_params_group_matmul_itr"
                        )
                        pklReplayFunction = (
                            "GroupMatmulTestCase.replay_from_pickle"
                        )
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise
                return

            return wrapper

        return hypothesis_params_group_matmul_itr_impl


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
        dilation2,
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

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
            stride,
            padding,
            conv_input,
            conv_weight,
            conv_bias,
            dilation,
            output_padding,
            conv_input3d,
            conv_weight3d,
            dilation2,
        ) = val
        self.createData(
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
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        stride_list=conv_stride_def,
        padding_list=conv_padding_def,
        conv_bs_Range=CONV_BS_RANGE,
        conv_c_Range=CONV_C_RANGE,
        conv_h_Range=CONV_H_RANGE,
        conv_wd_Range=CONV_WD_RANGE,
        conv_oc_Range=CONV_OC_RANGE,
        conv_kh_Range=CONV_KH_RANGE,
        conv_kw_Range=CONV_KW_RANGE,
        conv_dilation2_list=CONV_DILATION2,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)

        dtype = draw(st.sampled_from(dtype_list))
        hypStr += f"dtype_list=[{dtype!r}], "
        freeze = draw(st.sampled_from(freeze_list))
        hypStr += f"freeze_list=[{freeze}], "
        cpp_wrapper = draw(st.sampled_from(cpp_wrapper_opt_list))
        hypStr += f"cpp_wrapper_opt_list=[{cpp_wrapper}], "
        stride = draw(st.sampled_from(stride_list))
        hypStr += f"stride_list=[{stride}], "
        padding = draw(st.sampled_from(padding_list))
        hypStr += f"padding_list=[{padding}], "
        conv_bs = draw(st.integers(conv_bs_Range.get_min(), conv_bs_Range.get_max()))
        hypStr += f"conv_bs_Range=Range({conv_bs},{conv_bs}), "
        conv_c = draw(st.integers(conv_c_Range.get_min(), conv_c_Range.get_max()))
        hypStr += f"conv_c_Range=Range({conv_c},{conv_c}), "
        conv_h = draw(st.integers(conv_h_Range.get_min(), conv_h_Range.get_max()))
        hypStr += f"conv_h_Range=Range({conv_h},{conv_h}), "
        conv_wd = draw(st.integers(conv_wd_Range.get_min(), conv_wd_Range.get_max()))
        hypStr += f"conv_wd_Range=Range({conv_wd},{conv_wd}), "
        conv_oc = draw(st.integers(conv_oc_Range.get_min(), conv_oc_Range.get_max()))
        hypStr += f"conv_oc_Range=Range({conv_oc},{conv_oc}), "
        conv_kh = draw(st.integers(conv_kh_Range.get_min(), conv_kh_Range.get_max()))
        hypStr += f"conv_kh_Range=Range({conv_kh},{conv_kh}), "
        conv_kw = draw(st.integers(conv_kw_Range.get_min(), conv_kw_Range.get_max()))
        hypStr += f"conv_kw_Range=Range({conv_kw},{conv_kw}), "
        conv_dilation2 = draw(st.sampled_from(conv_dilation2_list))
        hypStr += f"conv_dilation2_list=[{conv_dilation2}], "

        torch_type = DataTypes.get_torch_type(dtype)

        conv_input = (
            torch.randn(conv_bs, conv_c, conv_h, conv_wd, generator=generator)
            .type(torch_type)
            .to(memory_format=torch.channels_last)
        )
        conv_weight = (
            torch.randn(conv_oc, conv_c, conv_kh, conv_kw, generator=generator)
            .type(torch_type)
            .to(memory_format=torch.channels_last)
        )
        conv_bias = torch.randn(conv_oc, generator=generator).type(torch_type)

        stride = stride
        padding = padding
        dilation = [1, 1]
        output_padding = [0, 0]

        conv_input3d = torch.randn(conv_bs, conv_c, conv_kh, generator=generator).type(
            torch_type
        )
        conv_weight3d = torch.randn(conv_oc, conv_c, conv_kh, generator=generator).type(
            torch_type
        )
        dilation2 = conv_dilation2

        return (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
            stride,
            padding,
            conv_input,
            conv_weight,
            conv_bias,
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
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        stride_list=conv_stride_def,
        padding_list=conv_padding_def,
        conv_bs_Range=CONV_BS_RANGE,
        conv_c_Range=CONV_C_RANGE,
        conv_h_Range=CONV_H_RANGE,
        conv_wd_Range=CONV_WD_RANGE,
        conv_oc_Range=CONV_OC_RANGE,
        conv_kh_Range=CONV_KH_RANGE,
        conv_kw_Range=CONV_KW_RANGE,
        conv_dilation2_list=CONV_DILATION2,
        tensor_seed=0,
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
                verbosity=Verbosity.quiet,
            )
            @given(
                val=ConvTestCase.tensor_conv_strategy(
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    cpp_wrapper_opt_list=cpp_wrapper_opt_list,
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
                    tensor_seed=tensor_seed,
                ),
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    hypStr, tensor_seed, dtype, freeze, cpp_wrapper, stride, padding, *_ = val

                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_conv_itr called with invalid object"
                        )

                    obj.createDataFromVal(val)
                    test_args = {
                        "dtype": dtype,
                        "freeze_opt": freeze,
                        "stride": stride,
                        "padding": padding,
                        "cpp_wrapper": cpp_wrapper,
                    }

                    required_args = inspect.signature(function).parameters.keys()

                    function(
                        obj,
                        *args,
                        **{k: v for k, v in test_args.items() if k in required_args},
                        **kwargs,
                    )
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "ConvTestCase.hypothesis_params_conv_itr"
                        pklReplayFunction = "ConvTestCase.replay_from_pickle"
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise  # Re-raise the exception after printing
                return

            return wrapper

        return hypothesis_params_itr_impl


class EmbTestCase(Zentorch_TestCase):
    time_out = 10000
    max_example_per_test = 20

    def getData(self):
        return self.data

    def createData(
        self, dtype, R, W, k, embedding_matrix, emb_input, offsets, mlp_inputs
    ):
        self.data.create_data_emb(
            dtype=dtype,
            R=R,
            W=W,
            k=k,
            embedding_matrix=embedding_matrix,
            emb_input=emb_input,
            offsets=offsets,
            mlp_inputs=mlp_inputs,
        )

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
            mode,
            include_last_offset,
            sparse,
            scale_grad,
            R,
            W,
            k,
            embedding_matrix,
            emb_input,
            offsets,
            mlp_inputs,
        ) = val
        self.createData(
            dtype=dtype,
            R=R,
            W=W,
            k=k,
            embedding_matrix=embedding_matrix,
            emb_input=emb_input,
            offsets=offsets,
            mlp_inputs=mlp_inputs,
        )

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_emb_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        emb_rRange=EMB_R_RANGE,
        emb_wRange=EMB_W_RANGE,
        emb_dRange=EMB_D_RANGE,
        emb_mlp_list=EMB_MLP_OPT,
        mode_opt_list=MODE_OPT_DEF,
        include_last_offset_opt_list=INCLUDE_LAST_OFFSET_OPT_DEF,
        sparse_opt_list=SPARSE_OPT_DEF,
        scale_grad_opt_list=SCALE_GRAD_OPT_DEF,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)
        dtype = draw(st.sampled_from(dtype_list))
        hypStr += f"dtype_list=[{dtype!r}], "
        freeze = draw(st.sampled_from(freeze_list))
        hypStr += f"freeze_list=[{freeze}], "
        cpp_wrapper = draw(st.sampled_from(cpp_wrapper_opt_list))
        hypStr += f"cpp_wrapper_opt_list=[{cpp_wrapper}], "
        R = draw(st.integers(emb_rRange.get_min(), emb_rRange.get_max()))
        hypStr += f"emb_rRange=Range({R},{R}), "
        W = draw(st.integers(emb_wRange.get_min(), emb_wRange.get_max()))
        hypStr += f"emb_wRange=Range({W},{W}), "
        k = draw(st.integers(emb_dRange.get_min(), emb_dRange.get_max()))
        hypStr += f"emb_dRange=Range({k},{k}), "
        emb_mlp = draw(st.sampled_from(emb_mlp_list))
        hypStr += f"emb_mlp_list=[{emb_mlp}] "
        mode = draw(st.sampled_from(mode_opt_list))
        hypStr += f"mode=[{mode}] "
        include_last_offset = draw(st.sampled_from(include_last_offset_opt_list))
        hypStr += f"include_last_offset=[{include_last_offset}] "
        sparse = draw(st.sampled_from(sparse_opt_list))
        hypStr += f"sparse=[{sparse}] "
        scale_grad = draw(st.sampled_from(scale_grad_opt_list))
        hypStr += f"scale_grad=[{scale_grad}] "

        torch_type = DataTypes.get_torch_type(dtype)

        embedding_matrix = torch.randn(R, k, generator=generator).type(
            torch_type
        )  # Here K value holds the value from emb_d
        emb_input = torch.randint(0, R, (W,), generator=generator)
        offsets = torch.tensor([0, W])
        mlp_inputs = torch.randn(emb_mlp, k, generator=generator)

        return (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
            mode,
            include_last_offset,
            sparse,
            scale_grad,
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
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        emb_rRange=EMB_R_RANGE,
        emb_wRange=EMB_W_RANGE,
        emb_dRange=EMB_D_RANGE,
        emb_mlp_list=EMB_MLP_OPT,
        mode_opt_list=MODE_OPT_DEF,
        include_last_offset_opt_list=INCLUDE_LAST_OFFSET_OPT_DEF,
        sparse_opt_list=SPARSE_OPT_DEF,
        scale_grad_opt_list=SCALE_GRAD_OPT_DEF,
        tensor_seed=0,
        time_out=None,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "dtype_list is empty"

        def hypothesis_params_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=EmbTestCase.time_out if time_out is None else time_out,
                max_examples=EmbTestCase.max_example_per_test,
                verbosity=Verbosity.quiet,
            )
            @given(
                val=EmbTestCase.tensor_emb_strategy(
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    cpp_wrapper_opt_list=cpp_wrapper_opt_list,
                    emb_rRange=emb_rRange,
                    emb_wRange=emb_wRange,
                    emb_dRange=emb_dRange,
                    emb_mlp_list=emb_mlp_list,
                    mode_opt_list=mode_opt_list,
                    include_last_offset_opt_list=include_last_offset_opt_list,
                    sparse_opt_list=sparse_opt_list,
                    scale_grad_opt_list=scale_grad_opt_list,
                    tensor_seed=tensor_seed,
                ),
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_emb_itr called with invalid object"
                        )

                    (
                        hypStr,
                        tensor_seed,
                        dtype,
                        freeze,
                        cpp_wrapper,
                        mode,
                        include_last_offset,
                        sparse,
                        scale_grad,
                        *_,
                    ) = val

                    obj.createDataFromVal(val)

                    # Prepare the arguments to pass to the test function
                    test_args = {
                        "dtype": dtype,
                        "freeze_opt": freeze,
                        "mode": mode,
                        "include_last_offset": include_last_offset,
                        "sprs_opt": sparse,
                        "scale_opt": scale_grad,
                        "cpp_wrapper": cpp_wrapper,
                    }

                    # Get the required argument names for the test function
                    required_args = inspect.signature(function).parameters.keys()

                    # Call the test function with the appropriate arguments
                    function(
                        obj,
                        *args,
                        **{k: v for k, v in test_args.items() if k in required_args},
                        **kwargs,
                    )
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "EmbTestCase.hypothesis_params_emb_itr"
                        pklReplayFunction = "EmbTestCase.replay_from_pickle"
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise  # Re-raise the exception after printing
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
        input3d,
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

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
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

        self.createData(
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
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        bRange=B_RANGE,
        mRange=M_RANGE,
        kRange=K_RANGE,
        nRange=N_RANGE,
        mm_input_scaler_Range=MM_INPUT_SCALER_RANGE,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)

        dtype = draw(st.sampled_from(dtype_list))
        hypStr += f"dtype_list=[{dtype!r}], "
        freeze = draw(st.sampled_from(freeze_list))
        hypStr += f"freeze_list=[{freeze}], "
        cpp_wrapper = draw(st.sampled_from(cpp_wrapper_opt_list))
        hypStr += f"cpp_wrapper_opt_list=[{cpp_wrapper}], "
        b = draw(st.integers(bRange.get_min(), bRange.get_max()))
        hypStr += f"bRange=Range({b},{b}), "
        m = draw(st.integers(mRange.get_min(), mRange.get_max()))
        hypStr += f"mRange=Range({m},{m}), "
        k = draw(st.integers(kRange.get_min(), kRange.get_max()))
        hypStr += f"kRange=Range({k},{k}), "
        n = draw(st.integers(nRange.get_min(), nRange.get_max()))
        hypStr += f"nRange=Range({n},{n}), "
        mm_input_scalar = draw(
            st.integers(
                mm_input_scaler_Range.get_min(), mm_input_scaler_Range.get_max()
            )
        )

        torch_type = DataTypes.get_torch_type(dtype)

        x = torch.randn(m, k, generator=generator).type(torch_type)
        y = torch.randn(k, n, generator=generator).type(torch_type)
        result = torch.zeros(m, n).type(torch_type)

        input = torch.randn(m, n, generator=generator).type(torch_type)
        input1d = torch.randn(n, generator=generator).type(torch_type)

        if torch_type in [torch.bfloat16, torch.float32]:
            input_scalar = torch.rand((), generator=generator).type(torch_type)
        else:
            input_scalar = torch.randint(
                0, mm_input_scalar, (), generator=generator
            ).type(torch_type)

        empty_bias = torch.zeros(0).type(torch_type)
        result_m = torch.zeros(int(m)).type(torch_type)
        result_1 = torch.zeros(1).type(torch_type)

        A = torch.randn(m, 1, generator=generator).type(torch_type)
        B = torch.randn(1, m, generator=generator).type(torch_type)

        x3d = torch.randn(b, m, k, generator=generator).type(torch_type)
        y3d = torch.randn(b, k, n, generator=generator).type(torch_type)
        input3d = torch.randn(b, m, n, generator=generator).type(torch_type)

        return (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
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
        )

    @staticmethod
    def hypothesis_params_mm_itr(
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        bRange=B_RANGE,
        mRange=M_RANGE,
        kRange=K_RANGE,
        nRange=N_RANGE,
        mm_input_scaler_Range=MM_INPUT_SCALER_RANGE,
        tensor_seed=0,
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
                verbosity=Verbosity.quiet,
            )
            @given(
                val=MMTestCase.tensor_mm_strategy(
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    cpp_wrapper_opt_list=cpp_wrapper_opt_list,
                    bRange=bRange,
                    mRange=mRange,
                    kRange=kRange,
                    nRange=nRange,
                    mm_input_scaler_Range=mm_input_scaler_Range,
                    tensor_seed=tensor_seed,
                ),
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    hypStr, updated_tensor_seed, dtype, freeze, cpp_wrapper, *_ = val

                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_mm_itr called with invalid object"
                        )

                    obj.createDataFromVal(val)

                    test_args = {
                        "dtype": dtype,
                        "freeze_opt": freeze,
                        "cpp_wrapper": cpp_wrapper,
                    }

                    required_args = inspect.signature(function).parameters.keys()

                    function(
                        obj,
                        *args,
                        **{k: v for k, v in test_args.items() if k in required_args},
                        **kwargs,
                    )
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "MMTestCase.hypothesis_params_mm_itr"
                        pklReplayFunction = "MMTestCase.replay_from_pickle"
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise  # Re-raise the exception after printing
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
        batch,
        in_features,
        out_features,
        with_bias,
        dtype,
        group_size,
        woq_input,
        woq_weight,
        woq_bias,
        woq_mul_input,
        woq_add_input,
        woq_add_input_2,
        input_dim=2,
    ):
        self.data.create_data_woq(
            batch=batch,
            in_features=in_features,
            out_features=out_features,
            with_bias=with_bias,
            dtype=dtype,
            group_size=group_size,
            woq_input=woq_input,
            woq_weight=woq_weight,
            woq_bias=woq_bias,
            woq_mul_input=woq_mul_input,
            woq_add_input=woq_add_input,
            woq_add_input_2=woq_add_input_2,
            input_dim=input_dim,
        )

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            batch,
            in_features,
            out_features,
            with_bias,
            dtype,
            group_size,
            input_dim,
            freeze,
            cpp_wrapper,
            woq_input,
            woq_weight,
            woq_bias,
            woq_mul_input,
            woq_add_input,
            woq_add_input_2,
        ) = val

        self.createData(
            batch=batch,
            in_features=in_features,
            out_features=out_features,
            with_bias=with_bias,
            dtype=dtype,
            group_size=group_size,
            woq_input=woq_input,
            woq_weight=woq_weight,
            woq_bias=woq_bias,
            woq_mul_input=woq_mul_input,
            woq_add_input=woq_add_input,
            woq_add_input_2=woq_add_input_2,
            input_dim=input_dim,
        )

    @staticmethod
    def _woq_input_shape_for_dim(batch, in_features, input_dim, p=1, q=1):
        """Return an n-D input shape with ``in_features`` as the trailing dim.

        Mirrors ``x_for_qlinear`` in ``tensor_qlinear_strategy`` (which builds
        ``(m, k)`` / ``(m, p, k)`` / ``(m, p, q, k)``) — extra dims ``p`` and
        ``q`` are independent of ``batch``, so no divisibility constraint is
        required. ``p`` and ``q`` are ignored when ``input_dim`` does not need
        them.

            2-D = (batch, K)
            3-D = (batch, p, K)
            4-D = (batch, p, q, K)
        """
        if input_dim == 2:
            return (batch, in_features)
        if input_dim == 3:
            return (batch, p, in_features)
        if input_dim == 4:
            return (batch, p, q, in_features)
        raise ValueError(f"Unsupported input_dim: {input_dim}")

    @staticmethod
    @st.composite
    def tensor_woq_strategy(
        draw,
        batch_opt_list,
        in_features_opt_list,
        out_features_opt_list,
        bias_opt_list,
        dtype_opt_list=woq_dtypes,
        group_size_opt_list=woq_group_size_def,
        input_dim_opt_list=INPUT_DIM_OPT_DEF,
        freeze_list=freeze_def_opt,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        pRange=WOQ_X_RANGE,
        qRange=WOQ_Y_RANGE,
        tensor_seed=0,
    ):
        """Unified strategy for WOQ tests (per-channel and per-group).

        Generates all test data including dimensions and tensors.
        This is where ALL random tensor creation happens.

        Args:
            dtype_opt_list: List of dtype strings to test. Defaults to woq_dtype.
            group_size_opt_list: List of group sizes for per-group quantization.
                                 Defaults to woq_group_size_def ([16]). Pass [None] for per-channel.
            input_dim_opt_list: Ranks of ``woq_input`` (and the matching binary
                                inputs) to exercise. Defaults to ``INPUT_DIM_OPT_DEF``
                                (``[2]``) so existing callers keep producing 2-D
                                activations. Pass ``input_dim_opt`` (``[2, 3, 4]``)
                                to also cover rank-3/rank-4 inputs as seen in e.g.
                                vLLM's Whisper encoder.
            pRange/qRange: Integer ranges for the extra leading dims used when
                           ``input_dim >= 3``/``input_dim == 4`` respectively.
                           Mirrors ``pRange``/``qRange`` in
                           ``tensor_qlinear_strategy`` so input shapes are
                           ``(batch, K)`` / ``(batch, p, K)`` / ``(batch, p, q, K)``.
        """
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "

        # Draw dimensions
        batch = draw(st.sampled_from(batch_opt_list))
        hypStr += f"batch_opt_list=[{batch}], "
        in_features = draw(st.sampled_from(in_features_opt_list))
        hypStr += f"in_features_opt_list=[{in_features}], "
        out_features = draw(st.sampled_from(out_features_opt_list))
        hypStr += f"out_features_opt_list=[{out_features}], "
        with_bias = draw(st.sampled_from(bias_opt_list))
        hypStr += f"bias_opt_list=[{with_bias}], "

        # Draw dtype string and map to torch dtype
        dtype_str = draw(st.sampled_from(dtype_opt_list))
        hypStr += f"dtype_opt_list=[{dtype_str!r}], "
        dtype = DataTypes.get_torch_type(dtype_str)

        # Draw group_size
        group_size = draw(st.sampled_from(group_size_opt_list))
        if group_size is not None:
            hypStr += f"group_size_opt_list=[{group_size}], "

        # Draw input rank. Extra leading dims (p, q) are sampled independently
        # from ``batch`` — same convention as ``tensor_qlinear_strategy``
        input_dim = draw(st.sampled_from(input_dim_opt_list))
        hypStr += f"input_dim_opt_list=[{input_dim}], "
        freeze = draw(st.sampled_from(freeze_list))
        hypStr += f"freeze_list=[{freeze}], "
        cpp_wrapper = draw(st.sampled_from(cpp_wrapper_opt_list))
        hypStr += f"cpp_wrapper_opt_list=[{cpp_wrapper}], "
        p = draw(st.integers(pRange.get_min(), pRange.get_max())) if input_dim >= 3 else 1
        q = draw(st.integers(qRange.get_min(), qRange.get_max())) if input_dim == 4 else 1
        if input_dim >= 3:
            hypStr += f"pRange=Range({p}, {p}), "
        if input_dim == 4:
            hypStr += f"qRange=Range({q}, {q}), "

        # Create all tensors using seeded generator
        # This is the ONLY place where random tensors are created
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)

        input_shape = WOQTestCase._woq_input_shape_for_dim(
            batch, in_features, input_dim, p=p, q=q
        )
        output_shape = input_shape[:-1] + (out_features,)

        woq_input = torch.randn(*input_shape, dtype=dtype, generator=generator)
        woq_weight = torch.randn(out_features, in_features, dtype=dtype, generator=generator)
        woq_bias = (
            torch.randn(out_features, dtype=dtype, generator=generator)
            if with_bias
            else None
        )

        # For binary fusion tests (mul_add, add_add); built at the WOQ-linear
        # output shape so they broadcast cleanly against ``zentorch_result``.
        woq_mul_input = torch.randn(*output_shape, dtype=dtype, generator=generator)
        woq_add_input = torch.randn(*output_shape, dtype=dtype, generator=generator)
        woq_add_input_2 = torch.randn(*output_shape, dtype=dtype, generator=generator)

        return (
            hypStr,
            tensor_seed,
            batch,
            in_features,
            out_features,
            with_bias,
            dtype_str,
            group_size,
            input_dim,
            freeze,
            cpp_wrapper,
            woq_input,
            woq_weight,
            woq_bias,
            woq_mul_input,
            woq_add_input,
            woq_add_input_2,
        )

    @staticmethod
    def hypothesis_params_woq_itr(
        batch_opt_list,
        in_features_opt_list,
        out_features_opt_list,
        bias_opt_list,
        dtype_opt_list=woq_dtypes,
        group_size_opt_list=woq_group_size_def,
        input_dim_opt_list=INPUT_DIM_OPT_DEF,
        freeze_list=freeze_def_opt,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        pRange=WOQ_X_RANGE,
        qRange=WOQ_Y_RANGE,
        tensor_seed=0,
        time_out=None,
    ):
        """Unified hypothesis iterator for WOQ tests (per-channel and per-group).
        """

        def hypothesis_params_woq_itr_impl(function):
            @settings(
                deadline=WOQTestCase.time_out if time_out is None else time_out,
                max_examples=WOQTestCase.max_example_per_test,
                verbosity=Verbosity.quiet,
            )
            @given(
                val=WOQTestCase.tensor_woq_strategy(
                    batch_opt_list=batch_opt_list,
                    in_features_opt_list=in_features_opt_list,
                    out_features_opt_list=out_features_opt_list,
                    bias_opt_list=bias_opt_list,
                    dtype_opt_list=dtype_opt_list,
                    group_size_opt_list=group_size_opt_list,
                    input_dim_opt_list=input_dim_opt_list,
                    freeze_list=freeze_list,
                    cpp_wrapper_opt_list=cpp_wrapper_opt_list,
                    pRange=pRange,
                    qRange=qRange,
                    tensor_seed=tensor_seed,
                ),
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    hypStr, _, _, _, _, _, dtype, _, _, freeze, cpp_wrapper, *_ = val

                    obj.createDataFromVal(val)

                    test_args = {
                        "dtype": dtype,
                        "freeze_opt": freeze,
                        "cpp_wrapper": cpp_wrapper,
                    }

                    required_args = inspect.signature(function).parameters.keys()

                    # Call the test function with the appropriate arguments
                    function(
                        obj,
                        *args,
                        **{k: v for k, v in test_args.items() if k in required_args},
                        **kwargs,
                    )
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "WOQTestCase.hypothesis_params_woq_itr"
                        pklReplayFunction = "WOQTestCase.replay_from_pickle"
                        test_args = {
                            "dtype": dtype,
                            "freeze_opt": freeze,
                            "cpp_wrapper": cpp_wrapper,
                        }
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise
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
        input3d,
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
            input3d=input3d,
        )

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
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
            q_linear_output_dtype,
            q_linear_eltwise,
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
        self.createData(
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

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_qlinear_strategy(
        draw,
        input_dim_opt_list=INPUT_DIM_OPT_DEF,
        q_weight_list_opt_list=Q_WEIGHT_LIST_OPT_DEF,
        bias_opt_list=BIAS_OPT_DEF,
        q_granularity_opt_list=Q_GRANULARITY_OPT_DEF,
        q_zero_points_dtype_opt_list=Q_ZERO_POINTS_DTYPE_OPT_DEF,
        q_linear_dtype_opt_list=Q_LINEAR_DTYPE_OPT_DEF,
        q_linear_output_dtype_opt_list=Q_LINEAR_DTYPE_OPT_DEF,
        qlinear_eltwise_opt_list=QLINEAR_ELTWISE_OPT_DEF,
        dtype_list=qlinear_dtypes,
        freeze_list=freeze_def_opt,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        bRange=B_RANGE,
        mRange=M_RANGE,
        pRange=P_RANGE,
        qRange=Q_RANGE,
        kRange=K_RANGE,
        nRange=N_RANGE,
        matrix_dim_1_Range=MATRIX_DIM_1_RANGE,
        matrix_dim_2_Range=MATRIX_DIM_2_RANGE,
        matrix_dim_3_Range=MATRIX_DIM_3_RANGE,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)
        constants = HypothesisConstants()
        input_dim = draw(st.sampled_from(input_dim_opt_list))
        hypStr += f"input_dim_opt_list=[{input_dim}], "
        q_weight = draw(st.sampled_from(q_weight_list_opt_list))
        hypStr += f"q_weight_list_opt_list=[{q_weight}], "
        bias = draw(st.sampled_from(bias_opt_list))
        hypStr += f"bias_opt_list=[{bias}], "
        q_granularity = draw(st.sampled_from(q_granularity_opt_list))
        hypStr += f"q_granularity_opt_list=[{q_granularity!r}], "
        q_zero_points_dtype = draw(st.sampled_from(q_zero_points_dtype_opt_list))
        hypStr += f"q_zero_points_dtype_opt_list=[{q_zero_points_dtype!r}], "
        q_linear_dtype = draw(st.sampled_from(q_linear_dtype_opt_list))
        hypStr += f"q_linear_dtype_opt_list=[{q_linear_dtype!r}], "
        dtype = draw(st.sampled_from(dtype_list))
        hypStr += f"dtype_list=[{dtype!r}], "
        freeze = draw(st.sampled_from(freeze_list))
        hypStr += f"freeze_list=[{freeze}], "
        cpp_wrapper = draw(st.sampled_from(cpp_wrapper_opt_list))
        hypStr += f"cpp_wrapper_opt_list=[{cpp_wrapper}], "
        b = draw(st.integers(bRange.get_min(), bRange.get_max()))
        hypStr += f"bRange=Range({b}, {b}), "
        m = draw(st.integers(mRange.get_min(), mRange.get_max()))
        hypStr += f"mRange=Range({m}, {m}), "
        p = draw(st.integers(pRange.get_min(), pRange.get_max()))
        hypStr += f"pRange=Range({p}, {p}), "
        q = draw(st.integers(qRange.get_min(), qRange.get_max()))
        hypStr += f"qRange=Range({q}, {q}), "
        k = draw(st.integers(kRange.get_min(), kRange.get_max()))
        hypStr += f"kRange=Range({k}, {k}), "
        n = draw(st.integers(nRange.get_min(), nRange.get_max()))
        hypStr += f"nRange=Range({n}, {n}), "
        q_linear_output_dtype = draw(st.sampled_from(q_linear_output_dtype_opt_list))
        hypStr += f"q_linear_output_dtype=[{q_linear_output_dtype}], "
        q_linear_eltwise = draw(st.sampled_from(list(qlinear_eltwise_opt_list)))
        hypStr += f"q_linear_eltwise=[{q_linear_eltwise}], "
        matrix_dim_1 = draw(
            st.integers(matrix_dim_1_Range.get_min(), matrix_dim_1_Range.get_max())
        )
        hypStr += f"matrix_dim_1_Range=Range({matrix_dim_1}, {matrix_dim_1}), "
        matrix_dim_2 = draw(
            st.integers(matrix_dim_2_Range.get_min(), matrix_dim_2_Range.get_max())
        )
        hypStr += f"matrix_dim_2_Range=Range({matrix_dim_2}, {matrix_dim_2}), "
        matrix_dim_3 = draw(
            st.integers(matrix_dim_3_Range.get_min(), matrix_dim_3_Range.get_max())
        )
        hypStr += f"matrix_dim_3_Range=Range({matrix_dim_3}, {matrix_dim_3}), "

        torch_type = DataTypes.get_torch_type(dtype)

        y_int8_square = [
            torch.randint(
                constants.y_int8_min,
                constants.y_int8_max,
                (k, k),
                dtype=torch.int8,
                generator=generator,
            )
        ]
        bias_for_qlinear_square = [
            None,
            torch.randn(k, generator=generator).type(torch_type),
        ]

        # Scales will be divided with, so it can't be zero and recommended to be positive.
        y_scales_square = {
            "per_tensor": (1 + torch.abs(torch.randn((1,), generator=generator))).type(
                torch.float32
            ),
            "per_channel": (1 + torch.abs(torch.randn(k, generator=generator))).type(
                torch.float32
            ),
        }
        y_zero_points_square = {
            "per_tensor": torch.tensor(0).type(torch.int8),
            "per_channel": torch.zeros(k).type(torch.int8),
        }
        x_for_qlinear = {
            "float32": {
                2: torch.randn(m, k, generator=generator).type(torch.float32),
                3: torch.randn(m, p, k, generator=generator).type(torch.float32),
                4: torch.randn(m, p, q, k, generator=generator).type(torch.float32),
            },
            "bfloat16": {
                2: torch.randn(m, k, generator=generator).type(torch.bfloat16),
                3: torch.randn(m, p, k, generator=generator).type(torch.bfloat16),
                4: torch.randn(m, p, q, k, generator=generator).type(torch.bfloat16),
            },
            "int8": {
                2: torch.randint(
                    constants.y_int8_min,
                    constants.y_int8_max,
                    (m, k),
                    generator=generator,
                ).type(torch.int8),
                3: torch.randint(
                    constants.y_int8_min,
                    constants.y_int8_max,
                    (m, p, k),
                    generator=generator,
                ).type(torch.int8),
                4: torch.randint(
                    constants.y_int8_min,
                    constants.y_int8_max,
                    (m, p, q, k),
                    generator=generator,
                ).type(torch.int8),
            },
            "uint8": {
                2: torch.randint(
                    0, constants.zero_point_max, (m, k), generator=generator
                ).type(torch.uint8),
                3: torch.randint(
                    0, constants.zero_point_max, (m, p, k), generator=generator
                ).type(torch.uint8),
                4: torch.randint(
                    0, constants.zero_point_max, (m, p, q, k), generator=generator
                ).type(torch.uint8),
            },
        }
        y_int8 = [
            torch.randint(
                constants.y_int8_min, constants.y_int8_max, (k, n), generator=generator
            )
            .type(torch.int8)
            .t(),
            torch.randint(
                constants.y_int8_min, constants.y_int8_max, (n, k), generator=generator
            ).type(torch.int8),
        ]
        binary_input = {
            2: torch.randn(m, n, generator=generator, dtype=torch_type),
            3: torch.randn(m, p, n, generator=generator, dtype=torch_type),
            4: torch.randn(m, p, q, n, generator=generator, dtype=torch_type),
        }
        bias_for_qlinear = [
            None,
            torch.randn(n, generator=generator).type(torch_type),
        ]

        # Scales will be divided with, so it can't be zero and recommended to be positive.
        x_scales = {
            "per_tensor": (1 + torch.abs(torch.randn((1,), generator=generator))).type(
                torch.float32
            ),
        }
        x_zero_points = {
            "per_tensor": {
                "float32": {
                    "int8": torch.tensor(0).type(torch.int8),
                    "uint8": torch.randint(
                        0, constants.zero_point_max, (1,), generator=generator
                    ).type(torch.uint8),
                },
                "bfloat16": {
                    "int8": torch.tensor(0).type(torch.int8),
                    "uint8": torch.randint(
                        0, constants.zero_point_max, (1,), generator=generator
                    ).type(torch.uint8),
                },
                "int8": {
                    "int8": torch.zeros(1).type(torch.int8),
                    "uint8": torch.tensor(0).type(torch.int8),
                },
                "uint8": {
                    "int8": torch.randint(
                        0, constants.zero_point_max, (1,), generator=generator
                    ).type(torch.uint8),
                    "uint8": torch.randint(
                        0, constants.zero_point_max, (1,), generator=generator
                    ).type(torch.uint8),
                },
            },
        }

        # Scales will be divided with, so it can't be zero and recommended to be positive.
        y_scales = {
            "per_tensor": (1 + torch.abs(torch.randn((1,), generator=generator))).type(
                torch.float32
            ),
            "per_channel": (1 + torch.abs(torch.randn(n, generator=generator))).type(
                torch.float32
            ),
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
                    # Scales will be divided with, so it can't be zero and recommended to be positive.
                    "positive_scales": (
                        1 + torch.abs(torch.randn((1,), generator=generator))
                    ).type(torch.float32),
                },
                "int8": {
                    # Scales will be divided with, so it can't be zero and recommended to be positive.
                    "positive_scales": (
                        1 + torch.abs(torch.randn((1,), generator=generator))
                    ).type(torch.float32),
                },
            }
        }
        output_zero_points = {
            "per_tensor": {
                "float32": None,
                "bfloat16": None,
                "uint8": torch.randint(
                    0, constants.zero_point_max, (1,), generator=generator
                ).type(torch.uint8),
                "int8": torch.zeros(1).type(torch.int8),
            },
        }
        wrong_scales_per_channel = torch.randn(n + 1, generator=generator).type(
            torch.float32
        )
        wrong_zero_points_per_channel = torch.zeros(n + 1).type(torch.int8)
        y = torch.randn(k, n, generator=generator).type(torch_type)
        input1d = torch.randn(n, generator=generator).type(torch_type)
        x1 = [
            torch.randn(matrix_dim_1, matrix_dim_2, generator=generator).type(
                torch_type
            ),
            torch.randn(matrix_dim_2, matrix_dim_1, generator=generator)
            .transpose(0, 1)
            .type(torch_type),
        ]
        y1 = [
            torch.randn(matrix_dim_2, matrix_dim_3, generator=generator).type(
                torch_type
            ),
            torch.randn(matrix_dim_3, matrix_dim_2, generator=generator)
            .transpose(1, 0)
            .type(torch_type),
        ]
        x3d = torch.randn(b, m, k, generator=generator).type(torch_type)
        y3d = torch.randn(b, k, n, generator=generator).type(torch_type)
        input3d = torch.randn(b, m, n, generator=generator).type(torch_type)

        return (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
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
            q_linear_output_dtype,
            q_linear_eltwise,
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
        input_dim_opt_list=INPUT_DIM_OPT_DEF,
        q_weight_list_opt_list=Q_WEIGHT_LIST_OPT_DEF,
        bias_opt_list=BIAS_OPT_DEF,
        q_granularity_opt_list=Q_GRANULARITY_OPT_DEF,
        q_zero_points_dtype_opt_list=Q_ZERO_POINTS_DTYPE_OPT_DEF,
        q_linear_dtype_opt_list=Q_LINEAR_DTYPE_OPT_DEF,
        q_linear_output_dtype_opt_list=Q_LINEAR_DTYPE_OPT_DEF,
        dtype_list=qlinear_dtypes,
        qlinear_eltwise_opt_list=QLINEAR_ELTWISE_OPT_DEF,
        freeze_list=freeze_def_opt,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        bRange=B_RANGE,
        mRange=M_RANGE,
        pRange=P_RANGE,
        qRange=Q_RANGE,
        kRange=K_RANGE,
        nRange=N_RANGE,
        matrix_dim_1_Range=MATRIX_DIM_1_RANGE,
        matrix_dim_2_Range=MATRIX_DIM_2_RANGE,
        matrix_dim_3_Range=MATRIX_DIM_3_RANGE,
        time_out=None,
        tensor_seed=0,
    ):
        skip_reason = None
        if not all(
            [
                input_dim_opt_list,
                q_weight_list_opt_list,
                bias_opt_list,
                q_granularity_opt_list,
                q_zero_points_dtype_opt_list,
                q_linear_dtype_opt_list,
                dtype_list,
            ]
        ):
            skip_reason = "one or more required input lists are empty"

        def hypothesis_params_qlinear_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=QLinearTestCase.time_out if time_out is None else time_out,
                max_examples=QLinearTestCase.max_example_per_test,
                verbosity=Verbosity.quiet,
            )
            @given(
                val=QLinearTestCase.tensor_qlinear_strategy(
                    input_dim_opt_list,
                    q_weight_list_opt_list,
                    bias_opt_list,
                    q_granularity_opt_list,
                    q_zero_points_dtype_opt_list,
                    q_linear_dtype_opt_list,
                    q_linear_output_dtype_opt_list=q_linear_output_dtype_opt_list,
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    cpp_wrapper_opt_list=cpp_wrapper_opt_list,
                    qlinear_eltwise_opt_list=qlinear_eltwise_opt_list,
                    bRange=bRange,
                    mRange=mRange,
                    pRange=pRange,
                    qRange=qRange,
                    kRange=kRange,
                    nRange=nRange,
                    matrix_dim_1_Range=MATRIX_DIM_1_RANGE,
                    matrix_dim_2_Range=MATRIX_DIM_2_RANGE,
                    matrix_dim_3_Range=MATRIX_DIM_3_RANGE,
                    tensor_seed=tensor_seed,
                ),
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    (
                        hypStr,
                        tensor_seed,
                        dtype,
                        freeze,
                        cpp_wrapper,
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
                        q_linear_output_dtype,
                        q_linear_eltwise,
                        *_,
                    ) = val

                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_qlinear_itr called with invalid object"
                        )

                    obj.createDataFromVal(val)

                    test_args = {
                        "dtype": dtype,
                        "input_dim": input_dim,
                        "q_weight_idx": q_weight,
                        "bias_opt_idx": bias,
                        "q_granularity_val": q_granularity,
                        "q_zero_points_dtype": q_zero_points_dtype,
                        "input_dtype": q_linear_dtype,
                        "output_dtype": q_linear_output_dtype,
                        "eltwise_op": q_linear_eltwise,
                        "freeze_opt": freeze,
                        "cpp_wrapper": cpp_wrapper,
                    }

                    required_args = inspect.signature(function).parameters.keys()

                    function(
                        obj,
                        *args,
                        **{k: v for k, v in test_args.items() if k in required_args},
                        **kwargs,
                    )

                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "QLinearTestCase.hypothesis_params_qlinear_itr"
                        pklReplayFunction = "QLinearTestCase.replay_from_pickle"
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise  # Re-raise the exception after printing
                return

            return wrapper

        return hypothesis_params_qlinear_itr_impl


class SDPATestCase(Zentorch_TestCase):
    time_out = 20000  # TODO: Go back to 10000 JIRA ZENAI-3799
    max_example_per_test = 20

    def getData(self):
        return self.data

    def createData(self, dtype, sdpa_query, sdpa_key, sdpa_value, mask_shape):
        self.data.create_data_SDPA(
            dtype=dtype,
            sdpa_query=sdpa_query,
            sdpa_key=sdpa_key,
            sdpa_value=sdpa_value,
            mask_shape=mask_shape,
        )

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            cpp_wrapper,
            seq_length,
            batch_size,
            mask,
            num_heads,
            head_dim,
            sdpa_query,
            sdpa_key,
            sdpa_value,
            mask_shape,
        ) = val
        self.createData(
            dtype=dtype,
            sdpa_query=sdpa_query,
            sdpa_key=sdpa_key,
            sdpa_value=sdpa_value,
            mask_shape=mask_shape,
        )

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_sdpa_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        seq_length_opt_list=SEQ_LENGTH_OPT_DEF,
        batch_size_opt_list=BATCH_SIZE_OPT_DEF,
        mask_opt_list=MASK_OPT_DEF,
        num_heads_opt_list=NUM_HEADS_OPT_DEF,
        head_dim_opt_list=HEAD_DIM_OPT_DEF,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)
        dtype = draw(st.sampled_from(dtype_list))
        hypStr += f"dtype_list=[{dtype!r}], "
        cpp_wrapper = draw(st.sampled_from(cpp_wrapper_opt_list))
        hypStr += f"cpp_wrapper_opt_list=[{cpp_wrapper}], "
        seq_length = draw(st.sampled_from(seq_length_opt_list))
        hypStr += f"seq_length_opt_list=[{seq_length}], "
        batch_size = draw(st.sampled_from(batch_size_opt_list))
        hypStr += f"batch_size_opt_list=[{batch_size}], "
        mask = draw(st.sampled_from(mask_opt_list))
        hypStr += f"mask_opt_list=[{mask!r}], "
        num_heads = draw(st.sampled_from(num_heads_opt_list))
        hypStr += f"num_heads_opt_list=[{num_heads}], "
        head_dim = draw(st.sampled_from(head_dim_opt_list))
        hypStr += f"head_dim_opt_list=[{head_dim}], "

        torch_type = DataTypes.get_torch_type(dtype)

        sdpa_query = torch.randn(
            batch_size,
            num_heads,
            seq_length,
            head_dim,
            device="cpu",
            requires_grad=False,
        ).type(torch_type)

        sdpa_key = torch.randn(
            batch_size,
            num_heads,
            seq_length,
            head_dim,
            device="cpu",
            requires_grad=False,
        ).type(torch_type)

        sdpa_value = torch.randn(
            batch_size,
            num_heads,
            seq_length,
            head_dim,
            device="cpu",
            requires_grad=False,
        ).type(torch_type)

        mask_shape = (batch_size, num_heads, seq_length, seq_length)

        return (
            hypStr,
            tensor_seed,
            dtype,
            cpp_wrapper,
            seq_length,
            batch_size,
            mask,
            num_heads,
            head_dim,
            sdpa_query,
            sdpa_key,
            sdpa_value,
            mask_shape,
        )

    @staticmethod
    def hypothesis_params_sdpa_itr(
        dtype_list=supported_dtypes_def,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        seq_length_opt_list=SEQ_LENGTH_OPT_DEF,
        batch_size_opt_list=BATCH_SIZE_OPT_DEF,
        mask_opt_list=MASK_OPT_DEF,
        num_heads_opt_list=NUM_HEADS_OPT_DEF,
        head_dim_opt_list=HEAD_DIM_OPT_DEF,
        tensor_seed=0,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "dtype_list is empty"

        def hypothesis_params_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=SDPATestCase.time_out,
                max_examples=SDPATestCase.max_example_per_test,
                verbosity=Verbosity.quiet,
            )
            @given(
                val=SDPATestCase.tensor_sdpa_strategy(
                    dtype_list=dtype_list,
                    cpp_wrapper_opt_list=cpp_wrapper_opt_list,
                    seq_length_opt_list=seq_length_opt_list,
                    batch_size_opt_list=batch_size_opt_list,
                    mask_opt_list=mask_opt_list,
                    num_heads_opt_list=num_heads_opt_list,
                    head_dim_opt_list=head_dim_opt_list,
                    tensor_seed=tensor_seed,
                ),
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_sdpa_itr called with invalid object"
                        )

                    (
                        hypStr,
                        tensor_seed,
                        dtype,
                        cpp_wrapper,
                        seq_length,
                        batch_size,
                        mask,
                        num_heads,
                        head_dim,
                        sdpa_query,
                        sdpa_key,
                        sdpa_value,
                        mask_shape,
                        *_,
                    ) = val

                    obj.createDataFromVal(val)

                    # Prepare the arguments to pass to the test function
                    test_args = {
                        "dtype": dtype,
                        "mask_type": mask,
                        "head_dim": head_dim,
                        "cpp_wrapper": cpp_wrapper,
                    }

                    # Get the required argument names for the test function
                    required_args = inspect.signature(function).parameters.keys()

                    # Call the test function with the appropriate arguments
                    function(
                        obj,
                        *args,
                        **{k: v for k, v in test_args.items() if k in required_args},
                        **kwargs,
                    )
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "SDPATestCase.hypothesis_params_sdpa_itr"
                        pklReplayFunction = "SDPATestCase.replay_from_pickle"
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise  # Re-raise the exception after printing
                return

            return wrapper

        return hypothesis_params_itr_impl


class QuantEmbTestCase(Zentorch_TestCase):
    time_out = 10000
    max_example_per_test = 20

    def getData(self):
        return self.data

    def createData(
        self,
        dtype,
        num_embeddings,
        embedding_dim,
        num_bags,
        indices_size,
        weight,
        indices,
        offsets,
        scales,
        zero_points,
        packed_weight,
        cat_input,
    ):
        self.data.create_data_quant_emb(
            dtype=dtype,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_bags=num_bags,
            indices_size=indices_size,
            weight=weight,
            indices=indices,
            offsets=offsets,
            scales=scales,
            zero_points=zero_points,
            packed_weight=packed_weight,
            cat_input=cat_input,
        )

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
            num_embeddings,
            embedding_dim,
            num_bags,
            indices_size,
            include_last_offset,
            weight,
            indices,
            offsets,
            scales,
            zero_points,
            packed_weight,
            cat_input,
        ) = val
        self.createData(
            dtype=dtype,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_bags=num_bags,
            indices_size=indices_size,
            weight=weight,
            indices=indices,
            offsets=offsets,
            scales=scales,
            zero_points=zero_points,
            packed_weight=packed_weight,
            cat_input=cat_input,
        )

    @seed(seed=SEED)
    @staticmethod
    @st.composite
    def tensor_quant_emb_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        num_embeddings_range=QUANT_EMB_NUM_RANGE,
        embedding_dim_range=QUANT_EMB_D_RANGE,
        num_bags_range=EMB_NUM_OF_BAGS,
        indices_size_range=QUANT_EMB_W_RANGE,
        include_last_offset_opt_list=INCLUDE_LAST_OFFSET_OPT_DEF,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)

        dtype = draw(st.sampled_from(dtype_list))
        hypStr += f"dtype_list=[{dtype!r}], "
        freeze = draw(st.sampled_from(freeze_list))
        hypStr += f"freeze_list=[{freeze}], "
        cpp_wrapper = draw(st.sampled_from(cpp_wrapper_opt_list))
        hypStr += f"cpp_wrapper_opt_list=[{cpp_wrapper}], "

        num_embeddings = draw(st.integers(num_embeddings_range.get_min(), num_embeddings_range.get_max()))
        hypStr += f"num_embeddings_range=Range({num_embeddings},{num_embeddings}), "

        embedding_dim = draw(st.sampled_from(embedding_dim_range))
        hypStr += f"embedding_dim_range=[{embedding_dim}], "

        num_bags = draw(st.integers(num_bags_range.get_min(), num_bags_range.get_max()))
        hypStr += f"num_bags_range=Range({num_bags},{num_bags}), "

        indices_size = draw(st.integers(indices_size_range.get_min(), indices_size_range.get_max()))
        hypStr += f"indices_size_range=Range({indices_size},{indices_size}), "

        include_last_offset = draw(st.sampled_from(include_last_offset_opt_list))
        hypStr += f"include_last_offset_opt_list=[{include_last_offset}] "

        # Generate weight matrix (quantized to int4 range: 0-14)
        weight = torch.randint(low=0, high=15, size=(num_embeddings, embedding_dim), dtype=torch.int32, generator=generator)

        # Generate indices
        indices = torch.randint(0, num_embeddings, (indices_size,), dtype=torch.long, generator=generator)

        # Generate offsets based on num_bags
        if include_last_offset:
            offsets = torch.cat([
                torch.tensor([0], dtype=torch.long),
                torch.sort(torch.randint(1, indices_size, (num_bags - 1,), dtype=torch.long, generator=generator))[0],
                torch.tensor([indices_size], dtype=torch.long)
            ])
        else:
            offsets = torch.cat([
                torch.tensor([0], dtype=torch.long),
                torch.sort(torch.randint(1, indices_size, (num_bags - 1,), dtype=torch.long, generator=generator))[0]
            ])

        # Generate scales and zero_points for quantization
        scales = torch.rand(weight.size(0), 1, generator=generator).round(decimals=2)
        zero_points = torch.randint(low=0, high=16, size=(weight.size(0),), dtype=torch.int32, generator=generator)

        # Pack the weight (using AWQ int4 packing)
        # This will be done in the test itself, but we'll store None here
        packed_weight = None

        torch_type = DataTypes.get_torch_type(dtype)
        cat_input = torch.randn(num_bags, embedding_dim // 2, generator=generator).to(torch_type)

        return (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            cpp_wrapper,
            num_embeddings,
            embedding_dim,
            num_bags,
            indices_size,
            include_last_offset,
            weight,
            indices,
            offsets,
            scales,
            zero_points,
            packed_weight,
            cat_input,
        )

    @staticmethod
    def hypothesis_params_quant_emb_itr(
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        cpp_wrapper_opt_list=cpp_wrapper_def_opt,
        num_embeddings_range=QUANT_EMB_NUM_RANGE,
        embedding_dim_range=QUANT_EMB_D_RANGE,
        num_bags_range=EMB_NUM_OF_BAGS,
        indices_size_range=QUANT_EMB_W_RANGE,
        include_last_offset_opt_list=INCLUDE_LAST_OFFSET_OPT_DEF,
        tensor_seed=0,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "dtype_list is empty"

        def hypothesis_params_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            @settings(
                deadline=QuantEmbTestCase.time_out,
                max_examples=QuantEmbTestCase.max_example_per_test,
                verbosity=Verbosity.quiet,
            )
            @given(
                val=QuantEmbTestCase.tensor_quant_emb_strategy(
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    cpp_wrapper_opt_list=cpp_wrapper_opt_list,
                    num_embeddings_range=num_embeddings_range,
                    embedding_dim_range=embedding_dim_range,
                    num_bags_range=num_bags_range,
                    indices_size_range=indices_size_range,
                    include_last_offset_opt_list=include_last_offset_opt_list,
                    tensor_seed=tensor_seed,
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_quant_emb_itr called with invalid object"
                        )

                    (
                        hypStr,
                        tensor_seed,
                        dtype,
                        freeze,
                        cpp_wrapper,
                        num_embeddings,
                        embedding_dim,
                        num_bags,
                        indices_size,
                        include_last_offset,
                        *_,
                    ) = val

                    obj.createDataFromVal(val)

                    # Prepare the arguments to pass to the test function
                    test_args = {
                        "dtype": dtype,
                        "freeze_opt": freeze,
                        "cpp_wrapper": cpp_wrapper,
                        "include_last_offset": include_last_offset,
                    }

                    # Get the required argument names for the test function
                    required_args = inspect.signature(function).parameters.keys()

                    # Call the test function with the appropriate arguments
                    function(
                        obj,
                        *args,
                        **{k: v for k, v in test_args.items() if k in required_args},
                        **kwargs,
                    )
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "QuantEmbTestCase.hypothesis_params_quant_emb_itr"
                        pklReplayFunction = "QuantEmbTestCase.replay_from_pickle"
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise  # Re-raise the exception after printing
                return

            return wrapper

        return hypothesis_params_itr_impl


class RmsNormTestCase(Zentorch_TestCase):
    """Base class for RMS norm hypothesis-based tests.

    Provides a composite strategy and decorator for generating randomized
    (batch_size, hidden_size, dtype, freeze) combinations for testing
    zentorch_rms_norm and zentorch_add_rms_norm_ ops.
    """

    time_out = 10000
    max_example_per_test = 20

    def getData(self):
        return self.data

    def createData(
        self,
        dtype,
        batch_size,
        hidden_size,
        rms_input,
        rms_weight,
        rms_residual,
    ):
        self.data.create_data_rms_norm(
            dtype=dtype,
            batch_size=batch_size,
            hidden_size=hidden_size,
            rms_input=rms_input,
            rms_weight=rms_weight,
            rms_residual=rms_residual,
        )

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            batch_size,
            hidden_size,
            rms_input,
            rms_weight,
            rms_residual,
        ) = val
        self.createData(
            dtype=dtype,
            batch_size=batch_size,
            hidden_size=hidden_size,
            rms_input=rms_input,
            rms_weight=rms_weight,
            rms_residual=rms_residual,
        )

    @seed(seed=SEED)
    @staticmethod
    # The @st.composite decorator is used to define custom Hypothesis strategies
    # for generating complex test data structures.
    @st.composite
    def tensor_rms_norm_strategy(
        draw,
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        batch_size_Range=RMS_BATCH_SIZE_RANGE,
        hidden_size_Range=RMS_HIDDEN_SIZE_RANGE,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)

        dtype = draw(st.sampled_from(dtype_list))
        hypStr += f"dtype_list=[{dtype!r}], "
        freeze = draw(st.sampled_from(freeze_list))
        hypStr += f"freeze_list=[{freeze}], "
        batch_size = draw(
            st.integers(batch_size_Range.get_min(), batch_size_Range.get_max())
        )
        hypStr += f"batch_size_Range=Range({batch_size},{batch_size}), "
        hidden_size = draw(
            st.integers(hidden_size_Range.get_min(), hidden_size_Range.get_max())
        )
        hypStr += f"hidden_size_Range=Range({hidden_size},{hidden_size}), "

        torch_type = DataTypes.get_torch_type(dtype)

        # input tensor: (batch_size, hidden_size)
        rms_input = torch.randn(batch_size, hidden_size, generator=generator).type(
            torch_type
        )
        # weight tensor: (hidden_size,)
        rms_weight = torch.randn(hidden_size, generator=generator).type(torch_type)
        # residual tensor: (batch_size, hidden_size) — used by zentorch_add_rms_norm_
        rms_residual = torch.randn(batch_size, hidden_size, generator=generator).type(
            torch_type
        )

        return (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            batch_size,
            hidden_size,
            rms_input,
            rms_weight,
            rms_residual,
        )

    @staticmethod
    def hypothesis_params_rms_norm_itr(
        dtype_list=supported_dtypes_def,
        freeze_list=freeze_def_opt,
        batch_size_Range=RMS_BATCH_SIZE_RANGE,
        hidden_size_Range=RMS_HIDDEN_SIZE_RANGE,
        time_out=None,
        tensor_seed=0,
    ):
        skip_reason = None
        if not dtype_list:
            skip_reason = "dtype_list is empty"

        def hypothesis_params_rms_norm_itr_impl(function):
            if skip_reason:
                print(f"Skipping test - {function.__name__}: {skip_reason}")
                return unittest.skipIf(True, skip_reason)(function)

            # The @settings() decorator configures Hypothesis test parameters.
            @settings(
                deadline=RmsNormTestCase.time_out if time_out is None else time_out,
                max_examples=RmsNormTestCase.max_example_per_test,
                verbosity=Verbosity.quiet,
            )
            # The @given() decorator generates test inputs using the strategy.
            @given(
                val=RmsNormTestCase.tensor_rms_norm_strategy(
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    batch_size_Range=batch_size_Range,
                    hidden_size_Range=hidden_size_Range,
                    tensor_seed=tensor_seed,
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    (
                        hypStr,
                        tensor_seed,
                        dtype,
                        freeze,
                        batch_size,
                        hidden_size,
                        *_,
                    ) = val

                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_rms_norm_itr called with invalid object"
                        )

                    obj.createDataFromVal(val)

                    test_args = {
                        "dtype": dtype,
                        "freeze_opt": freeze,
                    }

                    required_args = inspect.signature(function).parameters.keys()

                    function(
                        obj,
                        *args,
                        **{k: v for k, v in test_args.items() if k in required_args},
                        **kwargs,
                    )
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "RmsNormTestCase.hypothesis_params_rms_norm_itr"
                        pklReplayFunction = "RmsNormTestCase.replay_from_pickle"
                        obj.handleException(
                            obj,
                            str(e),
                            hypStr,
                            function.__name__,
                            decName,
                            pklReplayFunction,
                            val,
                            test_args,
                        )
                    raise  # Re-raise the exception after printing
                return

            return wrapper

        return hypothesis_params_rms_norm_itr_impl
