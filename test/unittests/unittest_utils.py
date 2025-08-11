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
import pickle
from datetime import datetime

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
    Q_LINEAR_DTYPE_OPT_DEF,
    skip_test_pt_2_0,
    skip_test_pt_2_1,
    skip_test_pt_2_3,
    skip_test_pt_2_4,
    reset_dynamo,
    freeze_opt,
    freeze_def_opt,
    test_with_freeze_opt,
    woq_dtypes,
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
    conv_stride,
    conv_stride_def,
    conv_padding,
    conv_padding_def,
    at_ops,
    zt_ops,
    qlinear_eltwise_map,
    QLINEAR_ELTWISE_OPT_DEF,
    seq_length_opt,
    batch_size_opt,
    mask_type_opt,
    num_heads_opt,
    head_dim_opt,
    torch,
    DataTypes,
    SEED,

    # common variables
    B_RANGE,
    M_RANGE,
    K_RANGE,
    N_RANGE,
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
    EMB_D_RANGE,
    EMB_MLP_OPT,

    # mm vars
    MM_INPUT_SCALER_RANGE,

    # woq variables
    WOQ_M_RANGE,
    WOQ_X_RANGE,
    WOQ_Y_RANGE,
    WOQ_K_RANGE,
    woq_dtypes,
    WOQ_QZEROS_NONZERO_DIM_RANGE,

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
)


path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DUMP_ERRORS = os.getenv("ZENTORCH_UNITTEST_DUMP_ERROR_TENSORS", "0").lower() in ("1", "true", "yes")
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
        os.makedirs(os.path.join(path, "data"), exist_ok=True)
        if self.dump_errors:
            os.makedirs("error_dumps", exist_ok=True)
        self.data = Test_Data()

    def tearDown(self):
        del self.data
        shutil.rmtree(os.path.join(path, "data"))

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
        b = draw(st.integers(bRange.get_min(), bRange.get_max()))
        hypStr += f"bRange=Range({b},{b}), "
        m = draw(st.integers(mRange.get_min(), mRange.get_max()))
        hypStr += f"mRange=Range({m},{m}), "
        k = draw(st.integers(kRange.get_min(), kRange.get_max()))
        hypStr += f"kRange=Range({k},{k}), "
        n = draw(st.integers(nRange.get_min(), nRange.get_max()))
        hypStr += f"nRange=Range({n},{n}), "

        matrix_dim_1 = draw(
            st.integers(
                matrix_dim_1_Range.get_min(),
                matrix_dim_1_Range.get_max()
            )
        )
        hypStr += f"matrix_dim_1_Range=Range({matrix_dim_1},{matrix_dim_1}), "
        matrix_dim_2 = draw(
            st.integers(
                matrix_dim_2_Range.get_min(),
                matrix_dim_2_Range.get_max()
            )
        )
        hypStr += f"matrix_dim_2_Range=Range({matrix_dim_2},{matrix_dim_2}), "
        matrix_dim_3 = draw(
            st.integers(
                matrix_dim_3_Range.get_min(),
                matrix_dim_3_Range.get_max()
            )
        )
        hypStr += f"matrix_dim_3_Range=Range({matrix_dim_3},{matrix_dim_3}), "
        matrix_dim_4 = draw(
            st.integers(
                matrix_dim_4_Range.get_min(),
                matrix_dim_4_Range.get_max()
            )
        )
        hypStr += f"matrix_dim_4_Range=Range({matrix_dim_4},{matrix_dim_4}), "

        torch_type = DataTypes.get_torch_type(dtype)

        M = [
            torch.randn(matrix_dim_1, matrix_dim_3, generator=generator).type(torch_type),
            torch.randn(matrix_dim_3, generator=generator).type(torch_type),
        ]

        T1 = torch.randn(2, matrix_dim_3, matrix_dim_3, generator=generator).type(torch_type)

        x1 = [
            torch.randn(matrix_dim_1, matrix_dim_2, generator=generator).type(torch_type),
            torch.randn(matrix_dim_2, matrix_dim_1, generator=generator).transpose(0, 1).type(torch_type),
        ]

        y1 = [
            torch.randn(matrix_dim_2, matrix_dim_3, generator=generator).type(torch_type),
            torch.randn(matrix_dim_3, matrix_dim_2, generator=generator).transpose(1, 0).type(torch_type),
        ]

        M2 = torch.randn(matrix_dim_1, matrix_dim_3, matrix_dim_4, generator=generator).type(torch_type)

        M3 = torch.randn(matrix_dim_4, generator=generator).type(torch_type)

        x2 = [
            torch.randn(matrix_dim_1, matrix_dim_3, matrix_dim_2, generator=generator).type(torch_type),
            torch.randn(
                matrix_dim_1,
                matrix_dim_2,
                matrix_dim_3,
                generator=generator
            ).transpose(1, 2).type(torch_type),
            torch.randn(
                matrix_dim_3,
                matrix_dim_1,
                matrix_dim_2,
                generator=generator
            ).transpose(0, 1).type(torch_type),
        ]

        y2 = [
            torch.randn(matrix_dim_1, matrix_dim_2, matrix_dim_4, generator=generator).type(torch_type),
            torch.randn(
                matrix_dim_1,
                matrix_dim_4,
                matrix_dim_2,
                generator=generator
            ).transpose(1, 2).type(torch_type),
            torch.randn(
                matrix_dim_4,
                matrix_dim_2,
                matrix_dim_1,
                generator=generator
            ).transpose(0, 2).type(torch_type),
        ]

        x = torch.randn(m, k, generator=generator).type(torch_type)
        y = torch.randn(k, n, generator=generator).type(torch_type)
        x3d = torch.randn(b, m, k, generator=generator).type(torch_type)
        y3d = torch.randn(b, k, n, generator=generator).type(torch_type)
        input = torch.randn(m, n, generator=generator).type(torch_type)
        input3d = torch.randn(b, m, n, generator=generator).type(torch_type)

        return (
            hypStr,
            tensor_seed,
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
                    bRange=bRange,
                    mRange=mRange,
                    kRange=kRange,
                    nRange=nRange,
                    matrix_dim_1_Range=matrix_dim_1_Range,
                    matrix_dim_2_Range=matrix_dim_2_Range,
                    matrix_dim_3_Range=matrix_dim_3_Range,
                    matrix_dim_4_Range=matrix_dim_4_Range,
                    tensor_seed=tensor_seed,
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    (hypStr, tensor_seed, dtype, freeze, *_) = val

                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_addmm_itr called with invalid object"
                        )

                    obj.createDataFromVal(val)

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
            st.integers(
                mm_add_1D_m_Range.get_min(),
                mm_add_1D_m_Range.get_max()
            )
        )
        hypStr += f"mm_add_1D_m_Range=Range({mm_add_1D_m}, {mm_add_1D_m}), "
        mm_add_1D_k = draw(
            st.integers(
                mm_add_1D_k_Range.get_min(),
                mm_add_1D_k_Range.get_max()
            )
        )
        hypStr += f"mm_add_1D_k_Range=Range({mm_add_1D_k}, {mm_add_1D_k}), "
        mm_add_1D_n = draw(
            st.integers(
                mm_add_1D_n_Range.get_min(),
                mm_add_1D_n_Range.get_max()
            )
        )
        hypStr += f"mm_add_1D_n_Range=Range({mm_add_1D_n}, {mm_add_1D_n}), "
        mm_add_2D_m = draw(
            st.integers(
                mm_add_2D_m_Range.get_min(),
                mm_add_2D_m_Range.get_max()
            )
        )
        hypStr += f"mm_add_2D_m_Range=Range({mm_add_2D_m}, {mm_add_2D_m}), "
        mm_add_2D_k = draw(
            st.integers(
                mm_add_2D_k_Range.get_min(),
                mm_add_2D_k_Range.get_max()
            )
        )
        hypStr += f"mm_add_2D_k_Range=Range({mm_add_2D_k}, {mm_add_2D_k}), "
        mm_add_2D_n = draw(
            st.integers(
                mm_add_2D_n_Range.get_min(),
                mm_add_2D_n_Range.get_max()
            )
        )
        hypStr += f"mm_add_2D_n_Range=Range({mm_add_2D_n}, {mm_add_2D_n}), "
        mm_add_3D_m = draw(
            st.integers(
                mm_add_3D_m_Range.get_min(),
                mm_add_3D_m_Range.get_max()
            )
        )
        hypStr += f"mm_add_3D_m_Range=Range({mm_add_3D_m}, {mm_add_3D_m}), "
        mm_add_3D_k = draw(
            st.integers(
                mm_add_3D_k_Range.get_min(),
                mm_add_3D_k_Range.get_max()
            )
        )
        hypStr += f"mm_add_3D_k_Range=Range({mm_add_3D_k}, {mm_add_3D_k}), "
        mm_add_3D_n = draw(
            st.integers(
                mm_add_3D_n_Range.get_min(),
                mm_add_3D_n_Range.get_max()
            )
        )
        hypStr += f"mm_add_3D_n_Range=Range({mm_add_3D_n}, {mm_add_3D_n}), "
        mm_add_3D_p = draw(
            st.integers(
                mm_add_3D_p_Range.get_min(),
                mm_add_3D_p_Range.get_max()
            )
        )
        hypStr += f"mm_add_3D_p_Range=Range({mm_add_3D_p}, {mm_add_3D_p}), "
        mm_add_3D_q = draw(
            st.integers(
                mm_add_3D_q_Range.get_min(),
                mm_add_3D_q_Range.get_max()
            )
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
            torch.rand(mm_add_3D_p, mm_add_3D_q, mm_add_3D_n, generator=generator).type(torch_type),
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
                deadline=AddmmTestCase.time_out,
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

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
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

        conv_input = torch.randn(
            conv_bs,
            conv_c,
            conv_h,
            conv_wd,
            generator=generator
        ).type(torch_type).to(memory_format=torch.channels_last)
        conv_weight = torch.randn(
            conv_oc,
            conv_c,
            conv_kh,
            conv_kw,
            generator=generator
        ).type(torch_type).to(memory_format=torch.channels_last)
        conv_bias = torch.randn(conv_oc, generator=generator).type(torch_type)

        stride = stride
        padding = padding
        dilation = [1, 1]
        output_padding = [0, 0]

        conv_input3d = torch.randn(conv_bs, conv_c, conv_kh, generator=generator).type(torch_type)
        conv_weight3d = torch.randn(conv_oc, conv_c, conv_kh, generator=generator).type(torch_type)
        dilation2 = conv_dilation2

        return (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
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
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    (
                        hypStr,
                        tensor_seed,
                        dtype,
                        freeze,
                        stride,
                        padding,
                        *_) = val

                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_conv_itr called with invalid object"
                        )

                    obj.createDataFromVal(val)
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

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
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

        embedding_matrix = torch.randn(R, k, generator=generator).type(torch_type)   # Here K value holds the value from emb_d
        emb_input = torch.randint(0, R, (W,), generator=generator)
        offsets = torch.tensor([0, W])
        mlp_inputs = torch.randn(emb_mlp, k, generator=generator)

        return (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
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
                verbosity=Verbosity.quiet,
            )
            @given(
                val=EmbTestCase.tensor_emb_strategy(
                    dtype_list=dtype_list,
                    freeze_list=freeze_list,
                    emb_rRange=emb_rRange,
                    emb_wRange=emb_wRange,
                    emb_dRange=emb_dRange,
                    emb_mlp_list=emb_mlp_list,
                    mode_opt_list=mode_opt_list,
                    include_last_offset_opt_list=include_last_offset_opt_list,
                    sparse_opt_list=sparse_opt_list,
                    scale_grad_opt_list=scale_grad_opt_list,
                    tensor_seed=tensor_seed,
                )
            )
            def wrapper(
                obj,
                val,
                *args,
                **kwargs
            ):
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
                        mode,
                        include_last_offset,
                        sparse,
                        scale_grad,
                        *_) = val

                    obj.createDataFromVal(val)

                    # Prepare the arguments to pass to the test function
                    test_args = {
                        'dtype': dtype,
                        'freeze_opt': freeze,
                        'mode' : mode,
                        'include_last_offset' : include_last_offset,
                        'sprs_opt' : sparse,
                        'scale_opt' : scale_grad,
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

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
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
            input3d
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
                mm_input_scaler_Range.get_min(),
                mm_input_scaler_Range.get_max()
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
            input_scalar = torch.randint(0, mm_input_scalar, (), generator=generator).type(torch_type)

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
                    bRange=bRange,
                    mRange=mRange,
                    kRange=kRange,
                    nRange=nRange,
                    mm_input_scaler_Range=mm_input_scaler_Range,
                    tensor_seed=tensor_seed,
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    (hypStr, updated_tensor_seed, dtype, freeze, *_) = val

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

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            woq_m,
            woq_x,
            woq_y,
            woq_k,
            b,
            m,
            n,
            scales_dtype,
            woq_input_dim,
            woq_bias_idx,
            woq_qzeros_idx,
            group_size_val,
            woq_input,
            packing_ratio,
            woq_add,
            woq_mul,
            woq_qweight,
            woq_scales,
            woq_qzeros,
            woq_qzeros_nonzero,
            woq_bias,
            input3d,
            input1d,
        ) = val
        self.createData(
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
        woq_m_Range=WOQ_M_RANGE,
        woq_x_Range=WOQ_X_RANGE,
        woq_y_Range=WOQ_Y_RANGE,
        woq_k_Range=WOQ_K_RANGE,
        bRange=B_RANGE,
        mRange=M_RANGE,
        nRange=N_RANGE,
        woq_qzeros_nonzero_dim_Range=WOQ_QZEROS_NONZERO_DIM_RANGE,
        tensor_seed=0,
    ):
        hypStr = ""
        if not tensor_seed:
            tensor_seed = getRandomSeed()
        hypStr += f"tensor_seed={tensor_seed}, "
        generator = torch.Generator()
        generator.manual_seed(tensor_seed)
        dtype = draw(st.sampled_from(woq_dtypes_list))
        hypStr += f"woq_dtypes_list=[{dtype!r}], "
        woq_input_dim = draw(st.sampled_from(input_dim_opt_list))
        hypStr += f"input_dim_opt_list=[{woq_input_dim}], "
        woq_bias_idx = draw(st.sampled_from(bias_opt_list))
        hypStr += f"bias_opt_list=[{woq_bias_idx}], "
        woq_qzeros_idx = draw(st.sampled_from(woq_qzeros_opt_list))
        hypStr += f"woq_qzeros_opt_list=[{woq_qzeros_idx}], "
        scales_dtype = draw(st.sampled_from(scales_dtype_list))
        hypStr += f"scales_dtype_list=[{scales_dtype!r}], "
        freeze = draw(st.sampled_from(freeze_list))
        hypStr += f"freeze_list=[{freeze}], "
        group_size_val = draw(st.sampled_from(group_size_opt_list))
        hypStr += f"group_size_opt_list=[{group_size_val}], "
        woq_m = draw(st.integers(woq_m_Range.get_min(), woq_m_Range.get_max()))
        hypStr += f"woq_m_Range=Range({woq_m}, {woq_m}), "
        woq_x = draw(st.integers(woq_x_Range.get_min(), woq_x_Range.get_max()))
        hypStr += f"woq_x_Range=Range({woq_x}, {woq_x}), "
        woq_y = draw(st.integers(woq_y_Range.get_min(), woq_y_Range.get_max()))
        hypStr += f"woq_y_Range=Range({woq_y}, {woq_y}), "
        woq_k = draw(st.integers(woq_k_Range.get_min(), woq_k_Range.get_max()))
        hypStr += f"woq_k_Range=Range({woq_k}, {woq_k}), "
        b = draw(st.integers(bRange.get_min(), bRange.get_max()))
        hypStr += f"bRange=Range({b}, {b}), "
        m = draw(st.integers(mRange.get_min(), mRange.get_max()))
        hypStr += f"mRange=Range({m}, {m}), "
        n = draw(st.integers(nRange.get_min(), nRange.get_max()))
        hypStr += f"nRange=Range({n}, {n}), "
        woq_qzeros_nonzero_dim = draw(
            st.integers(
                woq_qzeros_nonzero_dim_Range.get_min(),
                woq_qzeros_nonzero_dim_Range.get_max()
            )
        )
        hypStr += f"woq_qzeros_nonzero_dim_Range=Range({woq_qzeros_nonzero_dim}, {woq_qzeros_nonzero_dim}), "

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
            2: torch.randn(woq_m, woq_k, generator=generator).type(torch_type),
            3: torch.randn(woq_m, woq_y, woq_k, generator=generator).type(torch_type),
            4: torch.randn(woq_m, woq_x, woq_y, woq_k, generator=generator).type(
                torch_type
            ),
        }
        woq_add = {
            2: torch.randn(woq_m, woq_n, generator=generator).type(torch_type),
            3: torch.randn(woq_m, woq_y, woq_n, generator=generator).type(torch_type),
            4: torch.randn(woq_m, woq_x, woq_y, woq_n, generator=generator).type(
                torch_type
            ),
        }
        woq_mul = {
            2: torch.randn(woq_m, woq_n, generator=generator).type(torch_type),
            3: torch.randn(woq_m, woq_y, woq_n, generator=generator).type(torch_type),
            4: torch.randn(woq_m, woq_x, woq_y, woq_n, generator=generator).type(
                torch_type
            ),
        }
        woq_qweight = torch.randn(
            woq_k, woq_n // packing_ratio, generator=generator
        ).type(torch.int32)
        woq_qweight = {
            "bfloat16": copy.deepcopy(woq_qweight),
            "float32": copy.deepcopy(woq_qweight),
        }
        woq_scales = torch.randn(woq_k // group_size, woq_n, generator=generator).type(
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
            (woq_k // group_size, woq_n // packing_ratio),
            generator=generator,
        ).type(torch.int32)
        woq_bias = [
            None,
            torch.randn(woq_n, generator=generator).type(torch_type),
        ]

        input3d = torch.randn(b, m, n, generator=generator).type(torch_type)
        input1d = torch.randn(n, generator=generator).type(torch_type)

        return (
            hypStr,
            tensor_seed,
            dtype,
            freeze,
            woq_m,
            woq_x,
            woq_y,
            woq_k,
            b,
            m,
            n,
            scales_dtype,
            woq_input_dim,
            woq_bias_idx,
            woq_qzeros_idx,
            group_size_val,
            woq_input,
            packing_ratio,
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
        woq_m_Range=WOQ_M_RANGE,
        woq_x_Range=WOQ_X_RANGE,
        woq_y_Range=WOQ_Y_RANGE,
        woq_k_Range=WOQ_K_RANGE,
        bRange=B_RANGE,
        mRange=M_RANGE,
        nRange=N_RANGE,
        woq_qzeros_nonzero_dim_Range=WOQ_QZEROS_NONZERO_DIM_RANGE,
        tensor_seed=0,
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
                verbosity=Verbosity.quiet,
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
                    woq_m_Range=woq_m_Range,
                    woq_x_Range=woq_x_Range,
                    woq_y_Range=woq_y_Range,
                    woq_k_Range=woq_k_Range,
                    bRange=bRange,
                    mRange=mRange,
                    nRange=nRange,
                    woq_qzeros_nonzero_dim_Range=woq_qzeros_nonzero_dim_Range,
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
                        woq_m,
                        woq_x,
                        woq_y,
                        woq_k,
                        b,
                        m,
                        n,
                        scales_dtype,
                        woq_input_dim,
                        woq_bias_idx,
                        woq_qzeros_idx,
                        group_size_val,
                        *_,
                    ) = val

                    if not hasattr(obj, "getData") or not isinstance(
                        obj.getData(), Test_Data
                    ):
                        raise RuntimeError(
                            "hypothesis_params_woq_itr called with invalid object"
                        )

                    obj.createDataFromVal(val)

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
                except Exception as e:
                    if not isinstance(e, unittest.SkipTest):
                        decName = "WOQTestCase.hypothesis_params_woq_itr"
                        pklReplayFunction = "WOQTestCase.replay_from_pickle"
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

    def createDataFromVal(self, val):
        (
            hypStr,
            tensor_seed,
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
            st.integers(
                matrix_dim_1_Range.get_min(),
                matrix_dim_1_Range.get_max()
            )
        )
        hypStr += f"matrix_dim_1_Range=Range({matrix_dim_1}, {matrix_dim_1}), "
        matrix_dim_2 = draw(
            st.integers(
                matrix_dim_2_Range.get_min(),
                matrix_dim_2_Range.get_max()
            )
        )
        hypStr += f"matrix_dim_2_Range=Range({matrix_dim_2}, {matrix_dim_2}), "
        matrix_dim_3 = draw(
            st.integers(
                matrix_dim_3_Range.get_min(),
                matrix_dim_3_Range.get_max()
            )
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
            torch.randn(k, generator=generator).type(torch_type)]
        y_scales_square = {
            "per_tensor": torch.randn((1,), generator=generator).type(torch.float32),
            "per_channel": torch.randn(k, generator=generator).type(torch.float32),
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
                    0,
                    constants.zero_point_max,
                    (m, k),
                    generator=generator
                ).type(torch.uint8),
                3: torch.randint(0, constants.zero_point_max, (
                    m,
                    p,
                    k
                ),
                    generator=generator
                ).type(torch.uint8),
                4: torch.randint(0, constants.zero_point_max, (
                    m,
                    p,
                    q,
                    k
                ), generator=generator
                ).type(torch.uint8),
            },
        }
        y_int8 = [
            torch.randint(
                constants.y_int8_min,
                constants.y_int8_max,
                (k, n), generator=generator).type(torch.int8).t(),
            torch.randint(
                constants.y_int8_min,
                constants.y_int8_max,
                (n, k), generator=generator).type(torch.int8),
        ]
        binary_input = {
            2: torch.randn(m, n, generator=generator),
            3: torch.randn(m, p, n, generator=generator),
            4: torch.randn(m, p, q, n, generator=generator),
        }
        bias_for_qlinear = [
            None,
            torch.randn(n, generator=generator).type(torch_type),
        ]
        x_scales = {
            "per_tensor": torch.randn((1,), generator=generator).type(torch.float32),
        }
        x_zero_points = {
            "per_tensor": {
                "float32": {
                    "int8": torch.tensor(0).type(torch.int8),
                    "uint8": torch.randint(
                        0,
                        constants.zero_point_max,
                        (1,),
                        generator=generator).type(torch.uint8),
                },
                "bfloat16": {
                    "int8": torch.tensor(0).type(torch.int8),
                    "uint8": torch.randint(
                        0,
                        constants.zero_point_max,
                        (1,), generator=generator).type(torch.uint8),
                },
                "int8": {
                    "int8": torch.zeros(1).type(torch.int8),
                    "uint8": torch.tensor(0).type(torch.int8),
                },
                "uint8": {
                    "int8": torch.randint(
                        0,
                        constants.zero_point_max,
                        (1,),
                        generator=generator).type(torch.uint8),
                    "uint8": torch.randint(
                        0,
                        constants.zero_point_max,
                        (1,),
                        generator=generator).type(torch.uint8),
                },
            },
        }
        y_scales = {
            "per_tensor": torch.randn((1,), generator=generator).type(torch.float32),
            "per_channel": torch.randn(n, generator=generator).type(torch.float32),
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
                    "positive_scales": torch.rand((1,), generator=generator).type(torch.float32),
                },
                "int8": {
                    "positive_scales": torch.rand((1,), generator=generator).type(torch.float32),
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
                    (1,),
                    generator=generator
                ).type(torch.uint8),
                "int8": torch.zeros(1).type(torch.int8),
            },
        }
        wrong_scales_per_channel = torch.randn(n + 1, generator=generator).type(torch.float32)
        wrong_zero_points_per_channel = torch.zeros(n + 1).type(torch.int8)
        y = torch.randn(k, n, generator=generator).type(torch_type)
        input1d = torch.randn(n, generator=generator).type(torch_type)
        x1 = [
            torch.randn(matrix_dim_1, matrix_dim_2, generator=generator).type(torch_type),
            torch.randn(
                matrix_dim_2,
                matrix_dim_1,
                generator=generator).transpose(0, 1).type(torch_type),
        ]
        y1 = [
            torch.randn(matrix_dim_2, matrix_dim_3, generator=generator).type(torch_type),
            torch.randn(
                matrix_dim_3,
                matrix_dim_2,
                generator=generator).transpose(1, 0).type(torch_type),
        ]
        x3d = torch.randn(b, m, k, generator=generator).type(torch_type)
        y3d = torch.randn(b, k, n, generator=generator).type(torch_type)
        input3d = torch.randn(b, m, n, generator=generator).type(torch_type)

        return (
            hypStr,
            tensor_seed,
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
                )
            )
            def wrapper(obj, val, *args, **kwargs):
                try:
                    (
                        hypStr,
                        tensor_seed,
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
