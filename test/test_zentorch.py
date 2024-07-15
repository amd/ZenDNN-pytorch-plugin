# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from torch.testing._internal.common_utils import TestCase, run_tests, SEED
import torch
import unittest
import copy
import torch.nn as nn
from importlib import metadata
from torch.fx.experimental.proxy_tensor import make_fx
from parameterized import parameterized
from itertools import product
from torch.torch_version import TorchVersion
from test_zentorch_llm import MaskedMHATest
import random

try:
    import zentorch

    # for pattern matcher
    from zentorch._utils import counters

    has_zentorch = True
except ImportError:
    has_zentorch = False

skip_test_pt_2_0 = False
skip_test_pt_2_3 = False

if torch.__version__[:3] == "2.0":
    skip_test_pt_2_0 = True

if torch.__version__[:3] < "2.3":
    skip_test_pt_2_3 = True

supported_dtypes = ["float32"]
if zentorch._C.is_bf16_supported():
    supported_dtypes.append("bfloat16")
else:
    print(
        "Warning: Skipping Bfloat16 Testcases since they "
        + "are not supported on this hardware"
    )

include_last_offset_opt = [True, False]
scale_grad_opt = [True, False]
mode_opt = [0, 1, 2]
sparse_opt = [True, False]


# when calling the torch.compile flow, we need inference_mode decorator
# that is not needed when invoking zentorch ops directly
# Checking _dynamo.reset is compatible with pytorch version
def reset_dynamo():
    if TorchVersion(torch.__version__) < "2.3":
        torch._dynamo.reset()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Test_Data(metaclass=Singleton):

    def create_data(self, dtype):
        torch_type = self.get_torch_type(dtype)
        self.b = torch.randint(1, 11, (1,)).item()
        self.m = torch.randint(1, 11, (1,)).item()
        self.k = torch.randint(1, 11, (1,)).item()
        self.n = torch.randint(1, 11, (1,)).item()

        # m*k, k*n, m*n
        self.x = torch.randn(self.m, self.k).type(torch_type)
        self.y = torch.randn(self.k, self.n).type(torch_type)

        self.input = torch.randn(self.m, self.n).type(torch_type)
        self.input1d = torch.randn(self.n).type(torch_type)

        self.empty_bias = torch.empty(0).type(torch_type)
        self.result_m = torch.empty(self.m).type(torch_type)
        self.result_1 = torch.empty(1).type(torch_type)

        self.A = torch.randn(self.m, 1).type(torch_type)
        self.B = torch.randn(1, self.m).type(torch_type)

        # b*m*k, b*k*n, b*m*n
        self.x3d = torch.randn(self.b, self.m, self.k).type(torch_type)
        self.y3d = torch.randn(self.b, self.k, self.n).type(torch_type)
        self.input3d = torch.randn(self.b, self.m, self.n).type(torch_type)

        self.R = torch.randint(11, 20, (1,)).item()
        self.W = torch.randint(1, 15, (1,)).item()
        self.embedding_matrix = torch.rand(self.R, 3).type(torch_type)
        self.emb_input = torch.randint(0, self.R, (self.W,))
        self.offsets = torch.tensor([0, self.W])
        self.mlp_inputs = torch.randn(2, self.k)

        self.M = [
            torch.randn(60, 30).type(torch_type),
            torch.randn(30).type(torch_type),
        ]

        self.x1 = [
            torch.randn(60, 40).type(torch_type),
            torch.randn(40, 60).transpose(0, 1).type(torch_type),
        ]

        self.y1 = [
            torch.randn(40, 30).type(torch_type),
            torch.randn(30, 40).transpose(1, 0).type(torch_type),
        ]

        self.M2 = torch.randn(60, 30, 50).type(torch_type)
        self.M3 = torch.randn(50).type(torch_type)

        self.x2 = [
            torch.randn(60, 30, 40).type(torch_type),
            torch.randn(60, 40, 30).transpose(1, 2).type(torch_type),
            torch.randn(30, 60, 40).transpose(0, 1).type(torch_type),
        ]

        self.y2 = [
            torch.randn(60, 40, 50).type(torch_type),
            torch.randn(60, 50, 40).transpose(1, 2).type(torch_type),
            torch.randn(50, 40, 60).transpose(0, 2).type(torch_type),
        ]

    def get_torch_type(self, str_type):
        dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "int": torch.int,
        }
        return dtypes[str_type]


class Zentorch_TestCase(TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        random.seed(SEED)
        self.data = Test_Data()

    def tearDown(self):
        del self.data


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_MM_OP(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_matmul_variants(self, dtype):
        self.data.create_data(dtype)
        # mm
        self.assertEqual(
            torch._C._VariableFunctions.mm(self.data.x, self.data.y),
            torch.ops.zentorch.zentorch_mm(self.data.x, self.data.y),
        )
        self.assertEqual(
            torch.matmul(self.data.x, self.data.y),
            torch.ops.zentorch.zentorch_mm(self.data.x, self.data.y),
        )
        self.assertEqual(
            torch.mm(self.data.x, self.data.y),
            torch.ops.zentorch.zentorch_mm(self.data.x, self.data.y),
        )

        self.assertEqual(
            self.data.x @ self.data.y,
            torch.ops.zentorch.zentorch_mm(self.data.x, self.data.y),
        )

        self.assertEqual(
            torch.mul(self.data.A, self.data.B),
            torch.ops.zentorch.zentorch_mm(self.data.A, self.data.B),
        )

    @parameterized.expand(supported_dtypes)
    def test_mm_mismatched_dimensions(self, dtype):
        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_mm(
                self.data.x,
                torch.reshape(
                    self.data.x,
                    (1, list(self.data.x.shape)[0], list(self.data.x.shape)[1]),
                ),
            )
        self.assertTrue(
            "zentorch_mm:  unsupported dims for self and mat2" in str(context.exception)
        )

    @parameterized.expand([("int",)])
    def test_mm_unsupported_dtype(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_mm(self.data.x, self.data.y)
        self.assertTrue(
            "zentorch_matmul: zentorch_matmul only supports Float and BFloat16"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_mm_relu(self, dtype):

        self.data.create_data(dtype)
        # mm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.mm(self.data.x, self.data.y)
            ),
            torch.ops.zentorch.zentorch_mm_relu(self.data.x, self.data.y),
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_ADDMM_OP(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_addmm_variants(self, dtype):

        self.data.create_data(dtype)
        # addmm
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input,
                self.data.x,
                self.data.y,
            ),
        )
        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y, beta=1.3
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y, beta=1.3
            ),
        )

        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y, alpha=1.3
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y, alpha=1.3
            ),
        )

        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y, alpha=1.3, beta=1.3
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y, alpha=1.3, beta=1.3
            ),
        )

        # addmm with 1-d input
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_addmm(
                    self.data.input1d, self.data.x, self.data.y, alpha=1.3, beta=1.3
                )
                self.assertTrue(
                    "zentorch_matmul: zentorch_matmul is not supported for "
                    "bf16 tensors when bias is defined and alpha is not equal "
                    "to 1" in str(context.exception)
                )
        else:
            self.assertEqual(
                torch._C._VariableFunctions.addmm(
                    self.data.input1d, self.data.x, self.data.y, alpha=1.3, beta=1.3
                ),
                torch.ops.zentorch.zentorch_addmm(
                    self.data.input1d, self.data.x, self.data.y, alpha=1.3, beta=1.3
                ),
            )

    @parameterized.expand(supported_dtypes)
    def test_addmm_mismatched_dimensions(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm(
                self.data.x,
                self.data.x,
                torch.reshape(
                    self.data.x,
                    (list(self.data.x.shape)[0], list(self.data.x.shape)[1], 1),
                ),
            )

        self.assertTrue(
            "zentorch_addmm:  unsupported dims for self, mat1 and mat2"
            in str(context.exception)
        )

    @parameterized.expand(["int"])
    def test_addmm_unsupported_dtype(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm(self.data.x, self.data.x, self.data.x)

        self.assertTrue(
            "zentorch_matmul: zentorch_matmul only supports Float and BFloat16"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_addmm_relu_with_kw(self, dtype):

        self.data.create_data(dtype)
        # addmm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    self.data.input, self.data.x, self.data.y, beta=1.5, alpha=1.7
                )
            ),
            torch.ops.zentorch.zentorch_addmm_relu(
                self.data.input, self.data.x, self.data.y, beta=1.5, alpha=1.7
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    self.data.input, self.data.x, self.data.y, alpha=1.7
                )
            ),
            torch.ops.zentorch.zentorch_addmm_relu(
                self.data.input, self.data.x, self.data.y, alpha=1.7
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    self.data.input, self.data.x, self.data.y, beta=1.5
                )
            ),
            torch.ops.zentorch.zentorch_addmm_relu(
                self.data.input, self.data.x, self.data.y, beta=1.5
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    self.data.input, self.data.x, self.data.y, beta=0.0
                )
            ),
            torch.ops.zentorch.zentorch_addmm_relu(
                self.data.input, self.data.x, self.data.y, beta=0.0
            ),
        )

    @parameterized.expand(supported_dtypes)
    def test_addmm_with_zero_alpha(self, dtype):

        self.data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y, alpha=0.0
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y, alpha=0.0
            ),
        )

    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_addmm_relu_without_kw(self, dtype):

        self.data.create_data(dtype)
        # addmm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    self.data.input, self.data.x, self.data.y
                )
            ),
            torch.ops.zentorch.zentorch_addmm_relu(
                self.data.input, self.data.x, self.data.y
            ),
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_BMM_OP(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_bmm_variants(self, dtype):

        self.data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.bmm(self.data.x3d, self.data.y3d),
            torch.ops.zentorch.zentorch_bmm(self.data.x3d, self.data.y3d),
        )

    @parameterized.expand(supported_dtypes)
    def test_bmm_unsupported_dims(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_bmm(self.data.x, self.data.y)

        self.assertTrue(
            "zentorch_bmm:  unsupported dims for self and mat2"
            in str(context.exception)
        )

    @parameterized.expand([("int",)])
    def test_bmm_unsupported_dtype(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_bmm(self.data.x3d, self.data.y3d)

        self.assertTrue(
            "zentorch_matmul: zentorch_matmul only supports Float and BFloat16"
            in str(context.exception)
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_BADDBMM_OP(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_baddbmm_variants(self, dtype):

        self.data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d
            ),
        )

    @parameterized.expand([("int",)])
    def test_baddbmm_unsupported_dtype(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d
            )

        self.assertTrue(
            "zentorch_matmul: zentorch_matmul only supports Float and BFloat16"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_baddbmm_unsupported_dims(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d.reshape((self.data.b * self.data.m), self.data.n),
                self.data.x3d,
                self.data.y3d,
            )

        self.assertTrue(
            "zentorch_baddbmm:  unsupported dims for self, batch1 and batch2"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_baddbmm_with_kw(self, dtype):
        self.data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, beta=1.4
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, beta=1.4
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4, beta=1.3
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4, beta=1.3
            ),
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_MATMUL_IMPL_OP(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    def test_zentorch_matmul_impl_for_mv_and_dot(self, dtype):

        self.data.create_data(dtype)
        # mv
        self.assertEqual(
            torch.mv(self.data.input, self.data.input1d),
            zentorch._C.zentorch_matmul_impl(
                self.data.input,
                self.data.input1d,
                self.data.empty_bias,
                self.data.result_m,
                [],
                [],
            ),
            atol=1e-3,
            rtol=1e-2,
        )
        # dot
        self.assertEqual(
            torch.dot(self.data.input1d, self.data.input1d),
            zentorch._C.zentorch_matmul_impl(
                self.data.input1d,
                self.data.input1d,
                self.data.empty_bias,
                self.data.result_1,
                [],
                [],
            ),
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TEST_EMBEDDING_BAG(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    def test_embedding_bag_zendnn(self, dtype):

        self.data.create_data(dtype)
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_embedding_bag(
                    self.data.embedding_matrix,
                    self.data.emb_input,
                    self.data.offsets,
                    False,
                    0,
                    False,
                    None,
                    False,
                    -1,
                )
            self.assertTrue(
                "Only fp32 type weights are supported in ZenDNN EmbeddingBag!"
                in str(context.exception)
            )

        else:
            y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(
                self.data.embedding_matrix,
                self.data.emb_input,
                self.data.offsets,
                False,
                0,
                False,
                None,
                False,
            )

            y_ebz, _, _, _ = torch.ops.zentorch.zentorch_embedding_bag(
                self.data.embedding_matrix,
                self.data.emb_input,
                self.data.offsets,
                False,
                0,
                False,
                None,
                False,
                -1,
            )

            self.assertEqual(y_eb, y_ebz)

    @parameterized.expand(
        product(
            supported_dtypes,
            mode_opt,
            include_last_offset_opt,
            sparse_opt,
            scale_grad_opt,
        )
    )
    def test_embedding_bag_sparse_scale_mode(
        self, dtype, mode, include_last_offset, sprs_opt, scale_opt
    ):

        self.data.create_data(dtype)

        # max mode is not supported whenever any of the sparse_opt
        # or scale_grad_opt is True
        y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(
            self.data.embedding_matrix,
            self.data.emb_input,
            self.data.offsets,
            scale_opt,
            mode,
            sprs_opt,
            None,
            include_last_offset,
        )
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_embedding_bag(
                    self.data.embedding_matrix,
                    self.data.emb_input,
                    self.data.offsets,
                    scale_opt,
                    mode,
                    sprs_opt,
                    None,
                    include_last_offset,
                    -1,
                )
            self.assertTrue(
                "Only fp32 type weights are supported in ZenDNN EmbeddingBag!"
                in str(context.exception)
            )
        else:
            y_ebz, _, _, _ = torch.ops.zentorch.zentorch_embedding_bag(
                self.data.embedding_matrix,
                self.data.emb_input,
                self.data.offsets,
                scale_opt,
                mode,
                sprs_opt,
                None,
                include_last_offset,
                -1,
            )

            self.assertEqual(y_eb, y_ebz)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_custom_embedding_bag_compile(self, dtype):

        new_dtype = self.data.get_torch_type(dtype)
        model = CustomModelEmbeddingBagNN(100, 10, dtype=new_dtype)
        input = torch.randint(0, 10000, (1, 10))
        model_output = model(input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(input)
        self.assertAlmostEqual(
            model_output.item(), compiled_graph_output.item(), places=6
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TEST_EMBEDDING(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    def test_embedding_zendnn(self, dtype):
        self.data.create_data(dtype)
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_embedding(
                    self.data.embedding_matrix, self.data.emb_input
                )
            self.assertTrue(
                "Only fp32 type weights are supported in ZenDNN Embedding!"
                in str(context.exception)
            )
        else:
            y_eb = torch._C._VariableFunctions.embedding(
                self.data.embedding_matrix, self.data.emb_input
            )

            y_ebz = torch.ops.zentorch.zentorch_embedding(
                self.data.embedding_matrix, self.data.emb_input
            )

            self.assertEqual(y_eb, y_ebz)

    @parameterized.expand(supported_dtypes)
    def test_embedding_sparse_scale(self, dtype):
        self.data.create_data(dtype)
        sparse_opt = [True, False]
        scale_grad_opt = [True, False]

        for sprs_opt in sparse_opt:
            for scale_opt in scale_grad_opt:
                if dtype == "bfloat16":
                    with self.assertRaises(RuntimeError) as context:
                        torch.ops.zentorch.zentorch_embedding(
                            self.data.embedding_matrix,
                            self.data.emb_input,
                            -1,
                            scale_opt,
                            sprs_opt,
                        )
                    self.assertTrue(
                        "Only fp32 type weights are supported in ZenDNN Embedding!"
                        in str(context.exception)
                    )
                else:
                    y_eb = torch._C._VariableFunctions.embedding(
                        self.data.embedding_matrix,
                        self.data.emb_input,
                        -1,
                        scale_opt,
                        sprs_opt,
                    )

                    y_ebz = torch.ops.zentorch.zentorch_embedding(
                        self.data.embedding_matrix,
                        self.data.emb_input,
                        -1,
                        scale_opt,
                        sprs_opt,
                    )

                    self.assertEqual(y_eb, y_ebz)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_custom_embedding_compile(self, dtype):
        new_dtype = self.data.get_torch_type(dtype)
        model = CustomModelEmbeddingNN(100, dtype=new_dtype)
        input = torch.randint(0, 10000, (10,))
        model_output = model(input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(input)
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TEST_EMBEDDING_BAG_GROUP(Zentorch_TestCase):

    @parameterized.expand(
        product(
            supported_dtypes,
            mode_opt,
            include_last_offset_opt,
            sparse_opt,
            scale_grad_opt,
        )
    )
    def test_embedding_bag_group_zendnn(
        self, dtype, mode, include_last_offset, sprs_opt, scale_opt
    ):

        self.data.create_data(dtype)
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_horizontal_embedding_bag_group(
                    [self.data.embedding_matrix] * 3,
                    [self.data.emb_input] * 3,
                    [self.data.offsets] * 3,
                    [scale_opt] * 3,
                    [mode] * 3,
                    [sprs_opt] * 3,
                    [None] * 3,
                    [include_last_offset] * 3,
                    [-1] * 3,
                )
            self.assertTrue(
                "Only fp32 type weights are supported in ZenDNN EmbeddingBag!"
                in str(context.exception)
            )

        else:
            y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(
                self.data.embedding_matrix,
                self.data.emb_input,
                self.data.offsets,
                scale_opt,
                mode,
                sprs_opt,
                None,
                include_last_offset,
            )

            y_ebz_list = torch.ops.zentorch.zentorch_horizontal_embedding_bag_group(
                [self.data.embedding_matrix] * 3,
                [self.data.emb_input] * 3,
                [self.data.offsets] * 3,
                [scale_opt] * 3,
                [mode] * 3,
                [sprs_opt] * 3,
                [None] * 3,
                [include_last_offset] * 3,
                [-1] * 3,
            )

            for i in range(0, int(len(y_ebz_list) / 4)):
                self.assertEqual(y_eb, y_ebz_list[i * 4])

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_embeddingbag(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN EmbeddingBag!"
            )

        self.data.create_data(dtype)
        model = CustomModelEmbeddingBagGroup(self.data.R)
        indices = self.data.emb_input
        offsets = self.data.offsets

        fx_g = make_fx(model)(indices, offsets)
        fx_g_output = fx_g(indices, offsets)
        fx_g_optimized = zentorch.optimize(fx_g)
        fx_g_optimized_output = fx_g_optimized(indices, offsets)

        self.assertEqual(fx_g_output, fx_g_optimized_output)

        target = torch.ops.zentorch.zentorch_horizontal_embedding_bag_group.default
        group_eb_count = 0

        for node in fx_g_optimized.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload) and node.target == target:
                group_eb_count += 1

        self.assertEqual(group_eb_count, 3)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_embeddingbag_compile(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN EmbeddingBag!"
            )

        self.data.create_data(dtype)
        model = CustomModelEmbeddingBagGroup(self.data.R)
        indices = self.data.emb_input
        offset = self.data.offsets

        native_output = model(indices, offset)
        reset_dynamo()

        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices, offset)

        self.assertEqual(native_output, compiled_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TEST_EMBEDDING_GROUP(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    def test_embedding_group_zendnn(self, dtype):

        self.data.create_data(dtype)
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_horizontal_embedding_group(
                    [self.data.embedding_matrix] * 3,
                    [self.data.emb_input] * 3,
                    [-1] * 3,
                    [False] * 3,
                    [False] * 3,
                )
            self.assertTrue(
                "Only fp32 type weights are supported in ZenDNN Embedding!"
                in str(context.exception)
            )

        else:
            y_eb = torch._C._VariableFunctions.embedding(
                self.data.embedding_matrix, self.data.emb_input
            )

            y_ebz_list = torch.ops.zentorch.zentorch_horizontal_embedding_group(
                [self.data.embedding_matrix] * 3,
                [self.data.emb_input] * 3,
                [-1] * 3,
                [False] * 3,
                [False] * 3,
            )

            for i in range(0, int(len(y_ebz_list))):
                self.assertEqual(y_eb, y_ebz_list[i])

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_embedding(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN Embedding!"
            )

        self.data.create_data(dtype)
        model = CustomModelEmbeddingGroup(self.data.R)
        x = self.data.emb_input

        fx_g = make_fx(model)(x)
        fx_g_output = fx_g(x)

        fx_g_optimized = zentorch.optimize(fx_g)

        fx_g_optimized_output = fx_g_optimized(x)

        self.assertEqual(fx_g_output, fx_g_optimized_output)

        target = torch.ops.zentorch.zentorch_horizontal_embedding_group.default
        group_eb_count = 0

        for node in fx_g_optimized.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload) and node.target == target:
                group_eb_count += 1

        self.assertEqual(group_eb_count, 3)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_embedding_compile(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN Embedding!"
            )

        self.data.create_data(dtype)
        model = CustomModelEmbeddingGroup(self.data.R)
        x = self.data.emb_input

        native_output = model(x)

        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(x)

        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_emb_and_embbag_common_node(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN Embedding and EmbeddingBag!"
            )

        self.data.create_data(dtype)
        model = CustomModel_Emb_EmbBag_Common_Node(self.data.R)
        indices = self.data.emb_input
        offsets = self.data.offsets

        native_output = model(indices, offsets)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices, offsets)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_emb_and_embbag_diff_node(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN Embedding and EmbeddingBag!"
            )

        self.data.create_data(dtype)
        model = CustomModel_Emb_EmbBag_Diff_Node(self.data.R)
        indices = self.data.emb_input
        offsets = self.data.offsets

        native_output = model(indices, offsets)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices, offsets)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_embedding_2d_inputs(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN Embedding!"
            )

        self.data.create_data(dtype)
        model = CustomModel_2D_Embedding(self.data.R)
        indices = torch.cat([torch.unsqueeze(self.data.emb_input, dim=0)] * 2)

        native_output = model(indices)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices)
        self.assertEqual(native_output, compiled_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TEST_HORIZONTAL_MLP(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_horizontal_mlp(self, dtype):

        self.data.create_data(dtype)
        model = Custom_Horizontal_GroupMLP_Model(self.data.get_torch_type(dtype))

        for i in range(len(self.data.x2)):
            for j in range(len(self.data.y2)):
                native_output = model(self.data.x2[i], self.data.y2[j])
                reset_dynamo()
                compiled_graph = torch.compile(model, backend="zentorch")

                compiled_output = compiled_graph(self.data.x2[i], self.data.y2[j])
                self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_horizontal_mlp_multi_user(self, dtype):

        self.data.create_data(dtype)
        model = Custom_Horizontal_GroupMLP_multi_user_Model(
            -1, self.data.get_torch_type(dtype)
        )
        for i in range(len(self.data.x2)):
            for j in range(len(self.data.y2)):
                native_output = model(self.data.x2[i], self.data.y2[j])
                reset_dynamo()
                compiled_graph = torch.compile(model, backend="zentorch")

                compiled_output = compiled_graph(self.data.x2[i], self.data.y2[j])
                self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    def test_horizontal_mlp_unsupported_dims_1(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_attn_horizontal_mlp_group(
                [self.data.x1[0]] * 3,
                [self.data.y1[0]] * 3,
                [self.data.y1[0]] * 3,
                [0.0] * 3,
                [1.0] * 3,
                [0] * 3,
                [1],
            )
        self.assertTrue(
            "zentorch_addmm: unsupported dims for self, mat1 and mat2"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_horizontal_mlp_unsupported_dims_2(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_attn_horizontal_mlp_group(
                [self.data.y2[0]] * 3,
                [self.data.y2[0]] * 3,
                [self.data.y2[0]] * 3,
                [0.0] * 3,
                [1.0] * 3,
                [0] * 3,
                [1],
            )
        self.assertTrue(
            "zentorch_addmm:  unsupported dims for self, mat1 and mat2"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_horizontal_mlp_input_shape_compatibility(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_attn_horizontal_mlp_group(
                [self.data.M[1]] * 3,
                [self.data.y1[0]] * 3,
                [self.data.y1[0]] * 3,
                [0.0] * 3,
                [1.0] * 3,
                [0] * 3,
                [1],
            )
        self.assertTrue(
            "input shape is incompatible with matrix multiplication "
            in str(context.exception)
        )

    @torch.inference_mode()
    def test_bf16_alpha_not_1(self):

        self.data.create_data("bfloat16")
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm(
                self.data.input1d, self.data.x, self.data.y, alpha=1.7
            )
            self.assertTrue(
                "zentorch_matmul: zentorch_matmul is not supported for bf16 \
                tensors when bias is defined and alpha is not equal to 1"
                in str(context.exception)
            )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TEST_GROUP_MLP(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_mlp_model(self, dtype):

        self.data.create_data(dtype)

        model = CustomModel_GroupMLP_Model(self.data.k, self.data.get_torch_type(dtype))

        native_output = model(self.data.x)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(self.data.x)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_mlp_model_relu(self, dtype):

        self.data.create_data(dtype)

        model = CustomModel_GroupMLP_Model_Relu(
            self.data.k, self.data.get_torch_type(dtype)
        )

        native_output = model(self.data.x)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(self.data.x)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_mlp_model_relu_gelu(self, dtype):

        self.data.create_data(dtype)

        model = CustomModel_GroupMLP_Model_Relu_Gelu(
            self.data.k, self.data.get_torch_type(dtype)
        )

        native_output = model(self.data.x)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(self.data.x)
        self.assertEqual(native_output, compiled_output, atol=1e-3, rtol=1e-5)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_incorrect_dims(self, dtype):

        self.data.create_data(dtype)

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm(self.data.x3d, self.data.x, self.data.x)
            self.assertTrue(
                "zentorch_addmm: unsupported dims for self, mat1 and mat2!"
                in str(context.exception)
            )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm_1dbias(
                self.data.x, self.data.x, self.data.x
            )
            self.assertTrue(
                "zentorch_addmm_1dbias: unsupported dims for self, mat1 and mat2!"
                in str(context.exception)
            )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.x, self.data.x3d, self.data.x3d
            )
            self.assertTrue(
                "zentorch_baddbmm:  unsupported dims for self, batch1 and batch2!"
                in str(context.exception)
            )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_mm(self.data.x3d, self.data.x3d)
            self.assertTrue(
                "zentorch_mm:  unsupported dims for self and mat2!"
                in str(context.exception)
            )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_bmm(self.data.x, self.data.x)
            self.assertTrue(
                "zentorch_bmm:  unsupported dims for self and mat2!"
                in str(context.exception)
            )

    @torch.inference_mode()
    def test_bf16_alpha_not_1(self):

        self.data.create_data("bfloat16")

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm(
                self.data.input1d, self.data.x, self.data.y, alpha=1.7
            )
            self.assertTrue(
                "zentorch_matmul: zentorch_matmul is not supported for bf16 \
                tensors when bias is defined and alpha is not equal to 1"
                in str(context.exception)
            )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TEST_GROUP_EB_MLP(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_eb_mlp_model(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN EmbeddingBag!"
            )

        self.data.create_data(dtype)

        indices = self.data.emb_input
        offsets = self.data.offsets
        mlp_inputs = self.data.mlp_inputs

        model = CustomModel_Group_EB_MLP_Model(self.data.R, self.data.k)

        native_output = model(indices, offsets, mlp_inputs)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices, offsets, mlp_inputs)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_mlp_eb_model(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN EmbeddingBag!"
            )

        self.data.create_data(dtype)

        indices = self.data.emb_input
        offsets = self.data.offsets
        mlp_inputs = self.data.mlp_inputs

        model = CustomModel_Group_MLP_EB_Model(self.data.R, self.data.k)

        native_output = model(indices, offsets, mlp_inputs)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices, offsets, mlp_inputs)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_eb_mlp_model_multiple_groups(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN EmbeddingBag!"
            )

        self.data.create_data(dtype)

        indices = self.data.emb_input
        offsets = self.data.offsets
        mlp_inputs = self.data.mlp_inputs

        model = CustomModel_Group_EB_MLP_Model_multiple_groups(self.data.R, self.data.k)
        native_output = model(indices, offsets, mlp_inputs)
        reset_dynamo()
        compiled_model = torch.compile(model, backend="zentorch")
        compiled_output = compiled_model(indices, offsets, mlp_inputs)
        self.assertEqual(native_output, compiled_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
# Testing revealed one of the corner cases where the common output node can
# have heterogeneous nodes like embedding1, embedding2, sum1, sum2, embedding3.
# To test the above scenario, the following testcases are added.
# Both the group ops are being tested here, with the heterogeneous op being sum
class TEST_GROUP_EMBED_OPS_WITH_SUM_OPS(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_eb_with_sum(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN Embedding!"
            )

        self.data.create_data(dtype)

        indices = self.data.emb_input
        offsets = self.data.offsets

        model = CustomModel_EmbeddingBag_Sum_nodes(self.data.R)

        native_output = model(indices, offsets)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices, offsets)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_embedding_with_sum(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN EmbeddingBag!"
            )

        self.data.create_data(dtype)

        indices = self.data.emb_input

        model = CustomModel_Embedding_Sum_nodes(self.data.R)

        native_output = model(indices)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices)
        self.assertEqual(native_output, compiled_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestBF16Device(TestCase):
    @unittest.skipIf(
        not zentorch._C.is_bf16_supported(), "CPU does not support AVX512 BF16."
    )
    def test_bf16_device(self):
        self.assertTrue(zentorch._C.is_bf16_supported(), "CPU supports AVX512 BF16.")


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModel_Group_EB_MLP_Model(nn.Module):
    def __init__(self, num_embeddings, k):
        super(CustomModel_Group_EB_MLP_Model, self).__init__()
        self.eb_bags_grp = [torch.nn.EmbeddingBag(num_embeddings, 3)] * 3
        self.mlp_0 = torch.nn.Linear(k, 12)
        self.mlp_1 = torch.nn.Linear(12, 6)
        self.mlp_2 = torch.nn.Linear(6, 3)

    def forward(self, eb_input, eb_offset, mlp_input):
        eb_grp_outputs = [self.eb_bags_grp[i](eb_input, eb_offset) for i in range(3)]
        mlp_output = self.mlp_0(mlp_input)
        mlp_output = self.mlp_1(mlp_output)
        mlp_output = self.mlp_2(mlp_output)

        outputs = eb_grp_outputs + [mlp_output]
        outputs = torch.cat(outputs, dim=1)

        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModel_Group_MLP_EB_Model(nn.Module):
    def __init__(self, num_embeddings, k):
        super(CustomModel_Group_MLP_EB_Model, self).__init__()
        self.eb_bags_grp = [torch.nn.EmbeddingBag(num_embeddings, 3)] * 3
        self.mlp_0 = torch.nn.Linear(k, 12)
        self.mlp_1 = torch.nn.Linear(12, 6)
        self.mlp_2 = torch.nn.Linear(6, 3)

    def forward(self, eb_input, eb_offset, mlp_input):
        mlp_output = self.mlp_0(mlp_input)
        mlp_output = self.mlp_1(mlp_output)
        mlp_output = self.mlp_2(mlp_output)

        eb_grp_outputs = [self.eb_bags_grp[i](eb_input, eb_offset) for i in range(3)]

        outputs = eb_grp_outputs + [mlp_output]
        outputs = torch.cat(outputs, dim=1)

        return outputs


class CustomModel_Group_EB_MLP_Model_multiple_groups(torch.nn.Module):
    def __init__(self, num_embeddings, k):
        super(CustomModel_Group_EB_MLP_Model_multiple_groups, self).__init__()
        # Common Nodes
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.eb_bags = [torch.nn.EmbeddingBag(num_embeddings, 3)] * 2

        self.bmlp_0 = torch.nn.Linear(k, 4)
        self.bmlp_1 = torch.nn.Linear(4, 4)
        self.bmlp_2 = torch.nn.Linear(4, 3)

        self.tmlp_0 = torch.nn.Linear(12, 4)
        self.tmlp_1 = torch.nn.Linear(4, 2)
        self.tmlp_2 = torch.nn.Linear(2, 2)
        self.tmlp_3 = torch.nn.Linear(2, 1)

    def forward(self, eb_inputs, eb_offsets, mlp_inputs):

        outputs = []

        for _ in range(3):
            eb_outputs = [eb_op(eb_inputs, eb_offsets) for eb_op in self.eb_bags]

            mlp_outputs = self.bmlp_0(mlp_inputs)
            mlp_outputs = self.relu(mlp_outputs)
            mlp_outputs = self.bmlp_1(mlp_outputs)
            mlp_outputs = self.relu(mlp_outputs)
            mlp_outputs = self.bmlp_2(mlp_outputs)
            mlp_outputs = self.relu(mlp_outputs)

            interaction_input = eb_outputs + [mlp_outputs]
            interaction_output = torch.concat(interaction_input, dim=1)

            tmlp_input = torch.concat([mlp_outputs, interaction_output], dim=1)

            tmlp_outputs = self.tmlp_0(tmlp_input)
            tmlp_outputs = self.relu(tmlp_outputs)
            tmlp_outputs = self.tmlp_1(tmlp_outputs)
            tmlp_outputs = self.relu(tmlp_outputs)
            tmlp_outputs = self.tmlp_2(tmlp_outputs)
            tmlp_outputs = self.relu(tmlp_outputs)
            tmlp_outputs = self.tmlp_3(tmlp_outputs)
            tmlp_outputs = self.sigmoid(tmlp_outputs)

            outputs.append(tmlp_outputs)

        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Horizontal_GroupMLP_Model(nn.Module):
    def __init__(self, dtype):
        super(Custom_Horizontal_GroupMLP_Model, self).__init__()
        self.linear1 = nn.Linear(60, 50, dtype=dtype)
        self.linear2 = nn.Linear(50, 60, dtype=dtype)
        self.linear3 = nn.Linear(60, 50, dtype=dtype)

    def forward(self, batch1, batch2):
        bmm_output = torch.bmm(batch1, batch2)
        # add_output = torch.add(bmm_output, input)

        # Perform three separate view operations
        view1 = bmm_output.view(-1, 60)
        view2 = bmm_output.view(-1, 50)
        view3 = bmm_output.view(-1, 60)

        # Pass through linear layers
        linear1_output = self.linear1(view1)
        linear2_output = self.linear2(view2)
        linear3_output = self.linear3(view3)

        view4 = linear1_output.view(-1, 50)
        view5 = linear2_output.view(-1, 50)
        view6 = linear3_output.view(-1, 50)
        output = torch.cat(
            (view4, view5, view6),
        )

        return output


@unittest.skipIf(not has_zentorch, "PT PLUGIN is not installed")
class Custom_Horizontal_GroupMLP_multi_user_Model(nn.Module):
    def __init__(self, arg_1, dtype):
        super(Custom_Horizontal_GroupMLP_multi_user_Model, self).__init__()
        self.linear1 = nn.Linear(60, 50, dtype=dtype)
        self.linear2 = nn.Linear(50, 60, dtype=dtype)
        self.linear3 = nn.Linear(60, 50, dtype=dtype)
        self.arg1 = arg_1

    def forward(self, batch1, batch2):
        bmm_output = torch.bmm(batch1, batch2)

        # Perform three separate view operations
        view1 = bmm_output.view(self.arg1, 60)
        view2 = bmm_output.view(self.arg1, 50)
        view3 = bmm_output.view(self.arg1, 60)

        # Pass through linear layers
        linear1_output = self.linear1(view1)
        linear2_output = self.linear2(view2)
        linear3_output = self.linear3(view3)

        view4 = linear1_output.view(-1, 50)
        view5 = linear2_output.view(-1, 50)
        view6 = linear3_output.view(-1, 50)
        output = torch.cat(
            (view4, view5, view6),
        )

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModel_GroupMLP_Model(nn.Module):
    def __init__(self, k, dtype) -> None:
        super(CustomModel_GroupMLP_Model, self).__init__()
        self.mlp_0 = torch.nn.Linear(k, 512, dtype=dtype)
        self.mlp_1 = torch.nn.Linear(512, 256, dtype=dtype)
        self.mlp_2 = torch.nn.Linear(256, 64, dtype=dtype)

    def forward(self, inputs):
        outputs = self.mlp_0(inputs)
        outputs = self.mlp_1(outputs)
        outputs = self.mlp_2(outputs)

        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModel_GroupMLP_Model_Relu(nn.Module):
    def __init__(self, k, dtype) -> None:
        super(CustomModel_GroupMLP_Model_Relu, self).__init__()

        self.post_op = torch.nn.ReLU()

        self.bmlp_0 = torch.nn.Linear(k, 512, dtype=dtype)
        self.bmlp_1 = torch.nn.Linear(512, 256, dtype=dtype)
        self.bmlp_2 = torch.nn.Linear(256, 64, dtype=dtype)

        self.intermediate_activation = torch.nn.Sigmoid()

        self.tmlp_0 = torch.nn.Linear(64, 32, dtype=dtype)
        self.tmlp_1 = torch.nn.Linear(32, 16, dtype=dtype)
        self.tmlp_2 = torch.nn.Linear(16, 8, dtype=dtype)

    def forward(self, inputs):
        outputs = self.bmlp_0(inputs)
        outputs = self.post_op(outputs)
        outputs = self.bmlp_1(outputs)
        outputs = self.post_op(outputs)
        outputs = self.bmlp_2(outputs)
        outputs = self.post_op(outputs)

        outputs = self.intermediate_activation(outputs)

        outputs = self.tmlp_0(outputs)
        outputs = self.post_op(outputs)
        outputs = self.tmlp_1(outputs)
        outputs = self.post_op(outputs)
        outputs = self.tmlp_2(outputs)
        outputs = self.post_op(outputs)

        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModel_GroupMLP_Model_Relu_Gelu(nn.Module):
    def __init__(self, k, dtype) -> None:
        super(CustomModel_GroupMLP_Model_Relu_Gelu, self).__init__()

        self.post_op_1 = torch.nn.ReLU()
        self.post_op_2 = torch.nn.GELU()

        self.bmlp_0 = torch.nn.Linear(k, 512, dtype=dtype)
        self.bmlp_1 = torch.nn.Linear(512, 256, dtype=dtype)
        self.bmlp_2 = torch.nn.Linear(256, 64, dtype=dtype)

        self.intermediate_activation = torch.nn.Sigmoid()

        self.tmlp_0 = torch.nn.Linear(64, 32, dtype=dtype)
        self.tmlp_1 = torch.nn.Linear(32, 16, dtype=dtype)
        self.tmlp_2 = torch.nn.Linear(16, 8, dtype=dtype)

    def forward(self, inputs):
        outputs = self.bmlp_0(inputs)
        outputs = self.post_op_1(outputs)
        outputs = self.bmlp_1(outputs)
        outputs = self.post_op_2(outputs)
        outputs = self.bmlp_2(outputs)
        outputs = self.post_op_1(outputs)

        outputs = self.intermediate_activation(outputs)

        outputs = self.tmlp_0(outputs)
        outputs = self.post_op_2(outputs)
        outputs = self.tmlp_1(outputs)
        outputs = self.post_op_1(outputs)
        outputs = self.tmlp_2(outputs)
        outputs = self.post_op_2(outputs)

        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModelEmbeddingBagNN(nn.Module):
    def __init__(self, embedding_dim, output_dim, dtype=torch.float):
        super(CustomModelEmbeddingBagNN, self).__init__()
        self.embedding = nn.EmbeddingBag(10000, embedding_dim, dtype=dtype)
        self.intermediate = nn.Linear(embedding_dim, output_dim, dtype=dtype)
        self.output = nn.Linear(output_dim, 1, dtype=dtype)

    def forward(self, input):
        embed = self.embedding(input)
        intermediate = self.intermediate(embed)
        output = self.output(intermediate)
        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModelEmbeddingNN(nn.Module):
    def __init__(self, embedding_dim, dtype=torch.float):
        super(CustomModelEmbeddingNN, self).__init__()
        self.embedding = nn.Embedding(10000, embedding_dim, dtype=dtype)

    def forward(self, input):
        embed = self.embedding(input)
        return embed


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModel_Emb_EmbBag_Diff_Node(nn.Module):
    def __init__(self, num_embeddings):
        super(CustomModel_Emb_EmbBag_Diff_Node, self).__init__()
        self.eb_bags_grp = [
            torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum"),
            torch.nn.Embedding(num_embeddings, 3),
            torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum"),
            torch.nn.Embedding(num_embeddings, 3),
        ]

    def forward(self, eb_input, eb_offset):
        outputs_grp_0 = [
            self.eb_bags_grp[0](eb_input, eb_offset),
            self.eb_bags_grp[2](eb_input, eb_offset),
        ]
        outputs_grp_1 = [
            self.eb_bags_grp[1](eb_input),
            self.eb_bags_grp[3](eb_input),
        ]

        output_0 = torch.sum(torch.cat(outputs_grp_0), dim=0)
        output_1 = torch.sum(torch.cat(outputs_grp_1), dim=0)

        return torch.cat([output_0, output_1])


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModel_Emb_EmbBag_Common_Node(nn.Module):
    def __init__(self, num_embeddings):
        super(CustomModel_Emb_EmbBag_Common_Node, self).__init__()
        self.eb_bags_grp = [
            torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum"),
            torch.nn.Embedding(num_embeddings, 3),
            torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum"),
            torch.nn.Embedding(num_embeddings, 3),
        ]

    def forward(self, eb_input, eb_offset):
        outputs_grp = [
            self.eb_bags_grp[0](eb_input, eb_offset),
            self.eb_bags_grp[1](eb_input),
            self.eb_bags_grp[2](eb_input, eb_offset),
            self.eb_bags_grp[3](eb_input),
        ]

        output = torch.sum(torch.cat(outputs_grp), dim=0)

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModel_Embedding_Sum_nodes(nn.Module):
    def __init__(self, num_embeddings):
        super(CustomModel_Embedding_Sum_nodes, self).__init__()
        self.emebdding_grp = [torch.nn.Embedding(num_embeddings, 3) for _ in range(10)]

    def forward(self, inputs):
        outputs_grp = [op(inputs) for op in self.emebdding_grp]

        outputs_grp[3] = torch.sum(outputs_grp[3], dim=1, keepdim=True)
        outputs_grp[5] = torch.sum(outputs_grp[3], dim=1, keepdim=True)

        output = torch.sum(torch.cat(outputs_grp, dim=1), dim=0)

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModel_EmbeddingBag_Sum_nodes(nn.Module):
    def __init__(self, num_embeddings):
        super(CustomModel_EmbeddingBag_Sum_nodes, self).__init__()
        self.eb_bags_grp = [
            torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum") for _ in range(10)
        ]

    def forward(self, eb_input, eb_offset):
        outputs_grp = [op(eb_input, eb_offset) for op in self.eb_bags_grp]

        outputs_grp[5] = torch.sum(outputs_grp[5], dim=1, keepdim=True)
        outputs_grp[6] = torch.sum(outputs_grp[6], dim=1, keepdim=True)

        output = torch.sum(torch.cat(outputs_grp, dim=1), dim=0)

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModel_2D_Embedding(nn.Module):
    def __init__(self, num_embeddings):
        super(CustomModel_2D_Embedding, self).__init__()
        self.embedding_1 = torch.nn.Embedding(num_embeddings, 3)
        self.embedding_2 = torch.nn.Embedding(num_embeddings, 3)

    def forward(self, inputs):
        output = self.embedding_1(inputs) + self.embedding_2(inputs)

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModelEmbeddingBagGroup(nn.Module):
    def __init__(self, num_embeddings):
        super(CustomModelEmbeddingBagGroup, self).__init__()
        self.eb_bags_grp_0 = [torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum")] * 5
        self.eb_bags_grp_1 = [torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum")] * 10
        self.eb_bags_grp_2 = [torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum")] * 6

    def forward(self, eb_input, eb_offset):
        eb_outputs_grp_0 = [
            self.eb_bags_grp_0[i](eb_input, eb_offset) for i in range(5)
        ]
        concat_eb_tensors_0 = torch.cat(eb_outputs_grp_0)

        eb_outputs_grp_1 = [
            self.eb_bags_grp_1[i](eb_input, eb_offset) for i in range(10)
        ]
        concat_eb_tensors_1 = torch.cat(eb_outputs_grp_1)

        eb_outputs_grp_2 = [
            self.eb_bags_grp_2[i](eb_input, eb_offset) for i in range(6)
        ]
        concat_eb_tensors_2 = torch.cat(eb_outputs_grp_2)

        output = torch.cat(
            [concat_eb_tensors_0, concat_eb_tensors_1, concat_eb_tensors_2]
        )

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class CustomModelEmbeddingGroup(nn.Module):
    def __init__(self, num_embeddings):
        super(CustomModelEmbeddingGroup, self).__init__()
        self.e_bags_grp_0 = [torch.nn.Embedding(num_embeddings, 3)] * 5
        self.e_bags_grp_1 = [torch.nn.Embedding(num_embeddings, 3)] * 10
        self.e_bags_grp_2 = [torch.nn.Embedding(num_embeddings, 3)] * 6

    def forward(self, e_input):
        e_outputs_grp_0 = [self.e_bags_grp_0[i](e_input) for i in range(5)]
        e_sum_0 = torch.unsqueeze(torch.sum(torch.cat(e_outputs_grp_0), dim=0), dim=0)

        e_outputs_grp_1 = [self.e_bags_grp_1[i](e_input) for i in range(10)]
        e_sum_1 = torch.unsqueeze(torch.sum(torch.cat(e_outputs_grp_1), dim=0), dim=0)

        e_outputs_grp_2 = [self.e_bags_grp_2[i](e_input) for i in range(6)]
        e_sum_2 = torch.unsqueeze(torch.sum(torch.cat(e_outputs_grp_2), dim=0), dim=0)

        output = torch.cat([e_sum_0, e_sum_1, e_sum_2])

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestZenTorchVersion(TestCase):
    def test_zentorch_version(self):
        self.assertTrue(zentorch.__version__, metadata.version("zentorch"))


class CustomModelBMMAdd1(nn.Module):
    def __init__(self):
        super(CustomModelBMMAdd1, self).__init__()

    def forward(self, input, batch1, batch2):
        bmm_res = torch.bmm(batch1, batch2)
        add_res = torch.add(bmm_res, input)
        baddbmm_res = torch.baddbmm(add_res, batch1, batch2, beta=1.5, alpha=1.4)
        return baddbmm_res


class CustomModelBMM_Unsupport(nn.Module):
    def __init__(self):
        super(CustomModelBMM_Unsupport, self).__init__()

    def forward(self, input, batch1, batch2):
        bmm_res = torch.bmm(batch1, batch2)
        add_res = torch.add(bmm_res, input)
        return add_res


class CustomModelAddmmRelu2(nn.Module):
    def __init__(self):
        super(CustomModelAddmmRelu2, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        relu1_res = torch.relu(add_res)
        addmm_res = torch.addmm(relu1_res, batch1, batch2, beta=1.7, alpha=1.6)
        relu2_res = torch.relu(addmm_res)
        return relu2_res


class CustomModelAddmmReLU1(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomModelAddmmReLU1, self).__init__()

        # Linear layer (addmm operation)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Forward pass with addmm and ReLU fused
        return torch.relu(self.linear(x))


class CustomModelMMAdd1(nn.Module):
    def __init__(self):
        super(CustomModelMMAdd1, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        return add_res


class CustomModelMMRelu2(nn.Module):
    def __init__(self):
        super(CustomModelMMRelu2, self).__init__()

    def forward(self, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        relu_res = torch.relu(mm_res)
        return relu_res


class CustomModelMMReLU1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModelMMReLU1, self).__init__()

        # Linear layers (mm operation)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass with mm and ReLU fused
        x = torch.relu(self.linear1(x))
        return torch.relu(self.linear2(x))


class CustomModelAddmmGelu(nn.Module):
    def __init__(self):
        super(CustomModelAddmmGelu, self).__init__()
        self.gelu = nn.GELU(approximate="tanh")
        self.gelu2 = nn.GELU()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        GELU1_res = self.gelu(add_res)
        addmm_res = torch.addmm(GELU1_res, batch1, batch2)
        GELU2_res = self.gelu2(addmm_res)
        return GELU2_res


class CustomModelAddmmGeluTanh(nn.Module):
    def __init__(self):
        super(CustomModelAddmmGeluTanh, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        GELU_res = nn.functional.gelu(add_res, approximate="tanh")
        return GELU_res


class CustomModelAddmmGeluExact(nn.Module):
    def __init__(self):
        super(CustomModelAddmmGeluExact, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        GELU_res = nn.functional.gelu(add_res, approximate="none")
        return GELU_res


class CustomModelMM_View_Unary_OP(nn.Module):
    def __init__(self):
        super(CustomModelMM_View_Unary_OP, self).__init__()
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        add_res.view(-1, 4)
        GELU1_res = self.gelu(add_res)
        return GELU1_res


# The node being cloned will not always be previous node
# While removing clone op from graph we can encounter this scenario
class CustomModelMM_Diff_User_In_Btw(nn.Module):
    def __init__(self):
        super(CustomModelMM_Diff_User_In_Btw, self).__init__()

    def forward(self, input, batch1, batch2):
        mm = torch.mm(batch1, batch2)
        cln = torch.clone(input)
        res = torch.add(mm, cln)
        return res


class CustomModel_LinearSiLUMul(nn.Module):
    def __init__(self, data, bias):
        super(CustomModel_LinearSiLUMul, self).__init__()
        self.m = data.m
        self.n = data.n
        self.k = data.k
        self.linear_1 = torch.nn.Linear(self.n, self.k, bias=bias)
        self.linear_2 = torch.nn.Linear(self.n, self.k, bias=bias)
        self.silu = torch.nn.SiLU()

    def forward(self, inp):
        inp_shape = inp.shape
        inp_view = inp.view(inp_shape[0] * inp_shape[1], inp_shape[2])
        inp1_view = inp.view(inp_shape[0] * inp_shape[1], inp_shape[2])
        linear_silu = self.silu(self.linear_1(inp_view))
        linear_silu_view = linear_silu.view(inp_shape[0], self.m, self.k)
        linear = self.linear_2(inp1_view)
        linear_view = linear.view(inp_shape[0], self.m, self.k)

        return linear_silu_view * linear_view


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestMMRELU(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_mm_relu_optimize(self, dtype):

        self.data.create_data(dtype)
        model = CustomModelMMRelu2().eval()
        for i in range(len(self.data.x1)):
            for j in range(len(self.data.y1)):
                model_output = model(self.data.x1[i], self.data.y1[j])
                reset_dynamo()
                compiled_graph = torch.compile(model, backend="zentorch")
                compiled_graph_output = compiled_graph(self.data.x1[i], self.data.y1[j])
                self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zero_input_optimize(self, dtype):

        self.data.create_data(dtype)
        model = CustomModelMMRelu2().eval()
        model_output = model(self.data.x1[0] * 0, self.data.y1[0] * 0)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(self.data.x1[0] * 0, self.data.y1[0] * 0)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_negative_input_optimize(self, dtype):

        self.data.create_data(dtype)
        model = CustomModelMMRelu2().eval()
        model_output = model(self.data.x1[0] * -1, self.data.y1[0] * -1)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(
            self.data.x1[0] * -1, self.data.y1[0] * -1
        )
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_custom_mm_relu1(self, dtype):

        self.data.create_data(dtype)
        model = CustomModelMMReLU1(self.data.n, self.data.m, self.data.k).eval()
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(self.data.input)
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestMMADD(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_mm_add_optimize(self, dtype):

        self.data.create_data(dtype)
        model = CustomModelMMAdd1().eval()
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    reset_dynamo()
                    zentorch_model = copy.deepcopy(model)
                    inductor_graph = torch.compile(model, backend="inductor")
                    inductor_graph_output = inductor_graph(
                        inp, self.data.x1[i], self.data.y1[j]
                    )
                    reset_dynamo()
                    zentorch_graph = torch.compile(zentorch_model, backend="zentorch")
                    zentorch_graph_output = zentorch_graph(
                        inp, self.data.x1[i], self.data.y1[j]
                    )

                    self.assertEqual(inductor_graph_output, zentorch_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zero_input(self, dtype):

        self.data.create_data(dtype)
        model = CustomModelMMAdd1().eval()
        for inp in self.data.M:
            model_output = model(inp * 0, self.data.x1[0] * 0, self.data.y1[0] * 0)
            reset_dynamo()
            compiled_graph = torch.compile(model, backend="zentorch")
            compiled_graph_output = compiled_graph(
                inp * 0, self.data.x1[0] * 0, self.data.y1[0] * 0
            )
            self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_inf_input(self, dtype):

        self.data.create_data(dtype)
        model = CustomModelMMAdd1().eval()
        for inp in self.data.M:
            model_output = model(inp / 0, self.data.x1[0] / 0, self.data.y1[0] / 0)
            reset_dynamo()
            compiled_graph = torch.compile(model, backend="zentorch")
            compiled_graph_output = compiled_graph(
                inp / 0, self.data.x1[0] / 0, self.data.y1[0] / 0
            )
            self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_nan_input(self, dtype):

        self.data.create_data(dtype)
        model = CustomModelMMAdd1().eval()
        for inp in self.data.M:
            reset_dynamo()
            zentorch_model = copy.deepcopy(model)
            inductor_graph = torch.compile(model, backend="inductor")
            inductor_graph_output = inductor_graph(
                inp * float("nan"),
                self.data.x1[0] * float("nan"),
                self.data.y1[0] * float("nan"),
            )
            reset_dynamo()
            zentorch_graph = torch.compile(zentorch_model, backend="zentorch")
            zentorch_graph_output = zentorch_graph(
                inp * float("nan"),
                self.data.x1[0] * float("nan"),
                self.data.y1[0] * float("nan"),
            )
            self.assertEqual(inductor_graph_output, zentorch_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_identity_input_nan(self, dtype):

        self.data.create_data(dtype)
        if dtype == "bfloat16":
            self.skipTest("Skipping it since this testcase is not applicable for BF16.")
        model = CustomModelMMAdd1().eval()
        model_output = model(
            torch.eye(self.data.M[0].shape[0], self.data.M[0].shape[1]),
            self.data.x1[0] * float("nan"),
            self.data.y1[0] * float("nan"),
        )
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(
            torch.eye(self.data.M[0].shape[0], self.data.M[0].shape[1]),
            self.data.x1[0] * float("nan"),
            self.data.y1[0] * float("nan"),
        )
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestADDMM_GELU(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_addmm_gelu_tanh(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")

        self.data.create_data(dtype)
        model = CustomModelAddmmGeluTanh().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(
                        inp, self.data.x1[i], self.data.y1[j]
                    )
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_addmm_gelu_exact(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")

        self.data.create_data(dtype)
        model = CustomModelAddmmGeluExact().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(
                        inp, self.data.x1[i], self.data.y1[j]
                    )
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_addmm_gelu(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")

        self.data.create_data(dtype)
        model = CustomModelAddmmGelu().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(
                        inp, self.data.x1[i], self.data.y1[j]
                    )
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_addmm_view_gelu(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")

        self.data.create_data(dtype)
        model = CustomModelMM_View_Unary_OP().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(
                        inp, self.data.x1[i], self.data.y1[j]
                    )
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_mm_diff_user_in_btw(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")

        self.data.create_data(dtype)
        model = CustomModelMM_Diff_User_In_Btw().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(
                        inp, self.data.x1[i], self.data.y1[j]
                    )
                    self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestADDMM_RELU(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_addmm_relu(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")

        self.data.create_data(dtype)
        model = CustomModelAddmmRelu2().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(
                        inp, self.data.x1[i], self.data.y1[j]
                    )
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_custom_addmm_relu1(self, dtype):
        self.data.create_data(dtype)
        model = CustomModelAddmmReLU1(self.data.n, self.data.m).eval()
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(self.data.input)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    @unittest.skip("Nan and Inf giving non-deterministic output")
    def test_custom_addmm_relu1_with_nan_or_inf(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it since this testcase is not applicable for BF16.")

        self.data.create_data(dtype)
        model = CustomModelAddmmReLU1(self.data.n, self.data.m).eval()
        # Nan's output is non-deterministic. Skipping Nan
        # self.data.input[0][0] = float("nan")
        self.data.input[1][1] = float("inf")
        reset_dynamo()
        zentorch_model = copy.deepcopy(model)
        inductor_graph = torch.compile(model, backend="inductor")
        inductor_graph_output = inductor_graph(self.data.input)
        reset_dynamo()
        zentorch_graph = torch.compile(zentorch_model, backend="zentorch")
        zentorch_graph_output = zentorch_graph(self.data.input)
        self.assertEqual(inductor_graph_output, zentorch_graph_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestLinear_Relu(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_linear_relu(self, dtype):
        self.data.create_data(dtype)
        model = nn.Sequential(nn.Linear(self.data.n, self.data.m), nn.ReLU())
        if dtype == "bfloat16":
            model = model.bfloat16()
        fx_g = make_fx(model)(self.data.input)
        fx_g_modified = zentorch.optimize(fx_g)
        fx_g_output = fx_g(self.data.input)
        fx_g_modified_output = fx_g_modified(self.data.input)
        self.assertEqual(fx_g_output, fx_g_modified_output)
        for node in fx_g_modified.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                if node.target.name() in ["aten::addmm"]:
                    self.assertEqual(
                        node.target, torch.ops.zentorch.zentorch_addmm_1dbias
                    )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestLinear_Gelu(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_linear_gelu_tanh(self, dtype):

        self.data.create_data(dtype)
        model = nn.Sequential(
            nn.Linear(self.data.n, self.data.m), nn.GELU(approximate="tanh")
        )
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(self.data.input)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_linear_gelu_none(self, dtype):

        self.data.create_data(dtype)
        model = nn.Sequential(
            nn.Linear(self.data.n, self.data.m), nn.GELU(approximate="none")
        )
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(self.data.input)
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestLinear_SiLU(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_linear_with_bias_silu(self, dtype):
        self.data.create_data(dtype)
        model = nn.Sequential(nn.Linear(self.data.n, self.data.m, bias=True), nn.SiLU())
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(self.data.input)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_linear_without_bias_silu(self, dtype):
        self.data.create_data(dtype)
        model = nn.Sequential(
            nn.Linear(self.data.n, self.data.m, bias=False), nn.SiLU()
        )
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(self.data.input)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_mm_silu(self, dtype):
        self.data.create_data(dtype)
        native_output = torch.nn.functional.silu(torch.matmul(self.data.x, self.data.y))
        zentorch_output = torch.ops.zentorch.zentorch_mm_silu(self.data.x, self.data.y)

        self.assertEqual(native_output, zentorch_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_addmm_silu(self, dtype):
        self.data.create_data(dtype)
        native_output = torch.nn.functional.silu(
            torch.addmm(self.data.input, self.data.x, self.data.y)
        )
        zentorch_output = torch.ops.zentorch.zentorch_addmm_silu(
            self.data.input, self.data.x, self.data.y
        )

        self.assertEqual(native_output, zentorch_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestLinear_SiLU_Mul(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_mm_silu_mul(self, dtype):
        self.data.create_data(dtype)
        native_output = (
            torch.nn.functional.silu(torch.matmul(self.data.x, self.data.y))
            * self.data.input
        )
        zentorch_output = torch.ops.zentorch.zentorch_mm_silu_mul(
            self.data.x, self.data.y, self.data.input
        )

        self.assertEqual(native_output, zentorch_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_addmm_silu_mul(self, dtype):
        self.data.create_data(dtype)
        bias = self.data.input.clone()
        native_output = (
            torch.nn.functional.silu(torch.addmm(bias, self.data.x, self.data.y))
            * self.data.input
        )
        zentorch_output = torch.ops.zentorch.zentorch_addmm_silu_mul(
            bias, self.data.x, self.data.y, self.data.input
        )

        self.assertEqual(native_output, zentorch_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_linear_silu_mul_with_bias(self, dtype):
        self.data.create_data(dtype)
        model = CustomModel_LinearSiLUMul(self.data, bias=True)
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_input = self.data.input.view(1, self.data.m, self.data.n)
        model_output = model(model_input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 0
        )
        compiled_graph_output = compiled_graph(model_input)
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_silu_mul"], 1
        )
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_linear_silu_mul_without_bias(self, dtype):
        self.data.create_data(dtype)
        model = CustomModel_LinearSiLUMul(self.data, bias=False)
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_input = self.data.input.view(1, self.data.m, self.data.n)
        model_output = model(model_input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 0)
        compiled_graph_output = compiled_graph(model_input)
        self.assertEqual(counters["zentorch"]["pattern_matcher_mm_silu_mul"], 1)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_mm_silu_mul_mismatched_dimensions(self, dtype):
        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_mm_silu_mul(
                self.data.x,
                self.data.y,
                torch.reshape(
                    self.data.input,
                    (1, list(self.data.input.shape)[0], list(self.data.input.shape)[1]),
                ),
            )
        self.assertTrue(
            "zentorch_mm_silu_mul: unsupported dims for mat1, mat2 and mat3"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_mm_silu_mul_mismatched_sizes(self, dtype):
        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_mm_silu_mul(
                self.data.x, self.data.y, self.data.x
            )
        self.assertTrue(
            "zentorch_mm_silu_mul: unsupported sizes for mat1, mat2 and mat3"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_silu_mul_mismatched_dimensions(self, dtype):
        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm_silu_mul(
                self.data.input,
                self.data.x,
                self.data.y,
                torch.reshape(
                    self.data.input,
                    (1, list(self.data.input.shape)[0], list(self.data.input.shape)[1]),
                ),
            )
        self.assertTrue(
            "zentorch_addmm_silu_mul: unsupported dims for mat1, mat2 and mat3"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_silu_mul_mismatched_sizes(self, dtype):
        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm_silu_mul(
                self.data.input, self.data.x, self.data.y, self.data.x
            )
        self.assertTrue(
            "zentorch_addmm_silu_mul: unsupported sizes for mat1, mat2 and mat3"
            in str(context.exception)
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestBMMADD(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_bmm_baddbmm(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue with BF16 path.")

        self.data.create_data(dtype)
        model = CustomModelBMMAdd1().eval()
        for i in range(len(self.data.x2)):
            for j in range(len(self.data.y2)):
                model_output = model(self.data.M2, self.data.x2[i], self.data.y2[j])
                reset_dynamo()
                compiled_graph = torch.compile(model, backend="zentorch")
                compiled_graph_output = compiled_graph(
                    self.data.M2, self.data.x2[i], self.data.y2[j]
                )
                self.assertEqual(
                    model_output, compiled_graph_output, atol=1e-5, rtol=1e-3
                )

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zentorch_baddbmm_unsupport(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue with BF16 path.")

        self.data.create_data(dtype)
        model = CustomModelBMM_Unsupport().eval()
        model_output = model(self.data.M3, self.data.x2[0], self.data.y2[0])
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(
            self.data.M3, self.data.x2[0], self.data.y2[0]
        )
        self.assertEqual(model_output, compiled_graph_output, atol=1e-5, rtol=1e-3)


class GeluErfPattern(torch.nn.Module):
    def __init__(self):
        super(GeluErfPattern, self).__init__()

    def forward(self, input):
        mul_0 = torch.mul(input, 0.5)
        mul_1 = torch.mul(input, 0.7071067811865476)
        erf_0 = torch.erf(mul_1)
        add_0 = torch.add(erf_0, 1)
        mul_2 = torch.mul(mul_0, add_0)
        return mul_2


class BMMtoMM_Pattern_1(nn.Module):
    def __init__(self):
        super(BMMtoMM_Pattern_1, self).__init__()

    def forward(self, arg_0, arg_1):
        exp_0 = arg_0.expand(arg_0.size())
        exp_1 = arg_1.expand(arg_0.size(0), arg_1.size(0), arg_1.size(1))
        bmm_0 = torch.bmm(exp_0, exp_1)
        return bmm_0


class BMMtoMM_Pattern_2(nn.Module):
    def __init__(self):
        super(BMMtoMM_Pattern_2, self).__init__()

    def forward(self, arg_0, arg_1):
        # Expand and view arg_0_
        exp_0 = arg_0.expand(arg_0.size())
        view_0 = exp_0.view(arg_0.size())
        exp_1 = arg_1.expand(arg_0.size(0), arg_1.size(0), arg_1.size(1))
        bmm_0 = torch.bmm(view_0, exp_1)
        return bmm_0


# mm silu pattern
class AddmmSiLUMulPattern(torch.nn.Module):
    def __init__(self):
        super(AddmmSiLUMulPattern, self).__init__()

    def forward(self, inp_0, inp_1, inp_2, bias_0):
        view_0 = inp_0.view(inp_0.shape[0] * inp_0.shape[1], inp_0.shape[2])
        mm_0 = torch.ops.zentorch.zentorch_addmm_silu.default(bias_0, view_0, inp_1)
        view_1 = mm_0.view(inp_0.shape[0], inp_0.shape[1], inp_2.shape[-1])
        view_2 = inp_2.view(inp_0.shape[0], inp_0.shape[1], inp_2.shape[-1])
        mul_0 = torch.mul(view_1, view_2)
        return mul_0


# pattern matcher tests
@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class TestPatternMatcher(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    def test_addmm_silu_mul_replacement(self, dtype):
        decomp_mm_silu_model = AddmmSiLUMulPattern()
        model = decomp_mm_silu_model.to("cpu").eval()
        compiled_model = torch.compile(model, backend="zentorch")
        amp_enabled = True if dtype == "bfloat16" else False
        new_dtype = self.data.get_torch_type(dtype)
        inp_0 = torch.rand((2, 2, 11), dtype=new_dtype)
        inp_1 = torch.rand((11, 53), dtype=new_dtype)
        inp_2 = torch.rand((4, 53), dtype=new_dtype)
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_addmm_silu_mul"], 0)
        with torch.inference_mode(), torch.cpu.amp.autocast(enabled=amp_enabled):
            _ = compiled_model(inp_0, inp_1, inp_2, inp_2)
            # test for both dtypes, two separate tests will be run
            self.assertEqual(counters["zentorch"]["pattern_matcher_addmm_silu_mul"], 1)

    @parameterized.expand(supported_dtypes)
    def test_gelu_replacement(self, dtype):
        decomp_gelu_model = GeluErfPattern()
        model = decomp_gelu_model.to("cpu").eval()
        compiled_model = torch.compile(model, backend="zentorch")
        new_dtype = self.data.get_torch_type(dtype)
        inp = torch.empty((4, 11), dtype=new_dtype)
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_gelu"], 0)
        with torch.inference_mode():
            _ = compiled_model(inp)
            # test for both dtypes, two separate tests will be run
            self.assertEqual(counters["zentorch"]["pattern_matcher_gelu"], 1)

    def test_gelu_replacement_autocast(self):
        inp = torch.empty((5, 13))
        decomp_gelu_model = GeluErfPattern()
        model = decomp_gelu_model.to("cpu").eval()
        compiled_model = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_gelu"], 0)
        with torch.inference_mode(), torch.cpu.amp.autocast():
            _ = compiled_model(inp)
            self.assertEqual(counters["zentorch"]["pattern_matcher_gelu"], 1)

    @parameterized.expand(supported_dtypes)
    def test_bmm_to_mm_replacement(self, dtype):
        custom_expand_model = BMMtoMM_Pattern_1()
        new_dtype = self.data.get_torch_type(dtype)
        arg_0 = torch.empty((512, 1, 4096), dtype=new_dtype)
        arg_1 = torch.empty((4096, 4096), dtype=new_dtype)
        model = custom_expand_model.to("cpu").eval()
        native_output = model(arg_0, arg_1)
        reset_dynamo()
        compiled_model = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 0)
        with torch.inference_mode():
            zentorch_graph_output = compiled_model(arg_0, arg_1)
            self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 1)
            self.assertEqual(native_output, zentorch_graph_output)

    @parameterized.expand(supported_dtypes)
    def test_bmm_to_mm_pattern_2_replacement(self, dtype):
        custom_expand_model = BMMtoMM_Pattern_2()
        model = custom_expand_model.to("cpu").eval()
        new_dtype = self.data.get_torch_type(dtype)
        arg_0 = torch.empty((512, 1, 4096), dtype=new_dtype)
        arg_1 = torch.empty((4096, 4096), dtype=new_dtype)
        counters.clear()
        native_output = model(arg_0, arg_1)
        reset_dynamo()
        compiled_model = torch.compile(model, backend="zentorch")
        self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 0)
        with torch.inference_mode():
            zentorch_graph_output = compiled_model(arg_0, arg_1)
            self.assertEqual(counters["zentorch"]["pattern_matcher_bmm_to_mm"], 1)
            self.assertEqual(native_output, zentorch_graph_output)


# small testcase for rope, does not have all combinations
@unittest.skipIf(
    skip_test_pt_2_3, "Skipping test as OP support available from PyTorch 2.3"
)
class MiniRoPETester(TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        random.seed(SEED)
        self.max_seq_len = 512
        self.batch_size = 4
        self.seq_len = 32
        self.head_size = 256
        self.num_heads = 16
        self.hidden_size = self.head_size * self.num_heads

    def create_sinusoidal_positions(self, num_pos: int, dim: int) -> torch.Tensor:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        sinusoid_inp = torch.einsum(
            "i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq
        ).float()
        return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

    def test_rope(self):
        def _get_embed_positions(embed_positions, position_ids):
            if embed_positions.device != position_ids.device:
                embed_positions = embed_positions.to(position_ids.device)
                self.embed_positions = embed_positions
            return embed_positions.repeat(position_ids.shape[0], 1, 1)

        def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
            x = torch.stack((-x2, x1), dim=-1)
            return x.flatten(-2)

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(
            tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, offset: int = 1
        ) -> torch.Tensor:
            if offset == 1:
                sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
                cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
                return (tensor * cos) + (rotate_every_two(tensor) * sin)
            else:
                sin = sin[:, :, None, :].repeat(1, 1, 1, 2)
                cos = cos[:, :, None, :].repeat(1, 1, 1, 2)
                return (tensor * cos) + (rotate_half(tensor) * sin)

        def func(
            input,
            embed_positions,
            position_ids,
            num_heads,
            head_size,
            offset,
            rotary_dim,
        ):
            return torch.ops.zentorch.zentorch_rope(
                input,
                embed_positions,
                position_ids,
                num_heads,
                head_size,
                offset,
                rotary_dim,
            )

        def hf_forward(
            query, key, position_ids, embed_positions, offset=None, rotary_dim=None
        ):
            embed_positions = _get_embed_positions(embed_positions, position_ids)
            repeated_position_ids = position_ids.unsqueeze(-1).repeat(
                1, 1, embed_positions.shape[-1]
            )
            sincos = torch.gather(embed_positions, 1, repeated_position_ids)
            sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

            if rotary_dim < self.head_size:
                k_rot = key[:, :, :, :rotary_dim]
                k_pass = key[:, :, :, rotary_dim:]

                q_rot = query[:, :, :, :rotary_dim]
                q_pass = query[:, :, :, rotary_dim:]

                k_rot = apply_rotary_pos_emb(k_rot, sin, cos, offset)
                q_rot = apply_rotary_pos_emb(q_rot, sin, cos, offset)

                key = torch.cat([k_rot, k_pass], dim=-1)
                query = torch.cat([q_rot, q_pass], dim=-1)
            else:
                key = apply_rotary_pos_emb(key, sin, cos, offset)
                query = apply_rotary_pos_emb(query, sin, cos, offset)
            return query, key

        def upcast_tensors(a: torch.Tensor, b: torch.Tensor):
            # only two dtypes are supported at the moment - bf16 and fp32,
            # so we can get away with this shortcut approach
            if a.dtype == torch.float and b.dtype != torch.float:
                return a, b.to(torch.float)
            elif a.dtype != torch.float and b.dtype == torch.float:
                return a.to(torch.float), b
            else:
                return a, b

        kv_heads = [self.num_heads, self.num_heads // 2]
        dtypes = [torch.float32, torch.bfloat16]
        position_ids_t = torch.arange(self.seq_len).unsqueeze(0)
        position_ids_s = torch.Tensor([0]).to(torch.int64)
        model2rope_config = {
            "gptj": (64, 1, position_ids_t),
            "falcon": (self.head_size, 1, position_ids_s),
            "llama": (self.head_size, self.head_size // 2, position_ids_t),
            "gpt-neox": (24, 12, position_ids_t),
            "chatglm": (64, 1, position_ids_s),
            "codegen": (self.head_size, self.head_size // 2, position_ids_t),
        }
        for rope_config, kv_head, dtype in product(
            model2rope_config.values(), kv_heads, dtypes
        ):
            rotary_dim, offset, position_ids = rope_config
            # concat linear output
            linear_outs = torch.rand(
                self.batch_size,
                self.seq_len,
                self.hidden_size + kv_head * 2 * self.head_size,
            ).to(dtype)

            query = (
                linear_outs[:, :, : self.hidden_size]
                .contiguous()
                .view(self.batch_size, self.seq_len, self.num_heads, self.head_size)
            )
            key = (
                linear_outs[
                    :, :, self.hidden_size : self.hidden_size + kv_head * self.head_size
                ]
                .contiguous()
                .view(self.batch_size, self.seq_len, kv_head, self.head_size)
            )
            embed_positions = self.create_sinusoidal_positions(2048, rotary_dim)
            query_hf, key_hf = hf_forward(
                query, key, position_ids_t, embed_positions, offset, rotary_dim
            )
            # no concat q/k/v
            query_zentorch_no_concat, _, _ = torch.ops.zentorch.zentorch_rope(
                query,
                embed_positions,
                position_ids,
                self.num_heads,
                self.head_size,
                offset,
                rotary_dim,
            )
            key_zentorch_no_concat, _, _ = torch.ops.zentorch.zentorch_rope(
                key,
                embed_positions,
                position_ids,
                kv_head,
                self.head_size,
                offset,
                rotary_dim,
            )
            # concat q/k/v qkv_cocat -> ROPE -> (q, k, v)
            (
                query_zentorch,
                key_zentorch,
                value_zentorch,
            ) = torch.ops.zentorch.zentorch_rope(
                linear_outs,
                embed_positions,
                position_ids,
                self.num_heads,
                self.head_size,
                offset,
                rotary_dim,
            )

            # torch compile with zentorch backend.
            torch._dynamo.reset()
            func_compile = torch.compile(func, backend="zentorch")

            query_compile_no_concat, _, _ = func_compile(
                query,
                embed_positions,
                position_ids,
                self.num_heads,
                self.head_size,
                offset,
                rotary_dim,
            )
            query_compile, key_compile, value_compile = func_compile(
                linear_outs,
                embed_positions,
                position_ids,
                self.num_heads,
                self.head_size,
                offset,
                rotary_dim,
            )

            atol = 1e-5 if dtype == torch.float32 else 5e-3

            def upcast_and_assert(a: torch.Tensor, b: torch.Tensor, atol=1e-5):
                x, y = upcast_tensors(a, b)
                self.assertEqual(x, y, atol=atol, rtol=0)

            upcast_and_assert(query_compile_no_concat, query_hf, atol=atol)
            upcast_and_assert(query_compile, query_hf, atol=atol)
            upcast_and_assert(key_compile, key_hf, atol=atol)
            upcast_and_assert(query_hf, query_zentorch_no_concat, atol=atol)
            upcast_and_assert(key_hf, key_zentorch_no_concat, atol=atol)
            upcast_and_assert(query_hf, query_zentorch, atol=atol)
            upcast_and_assert(key_hf, key_zentorch, atol=atol)

            self.assertEqual(
                value_zentorch,
                linear_outs[:, :, self.hidden_size + kv_head * self.head_size :].view(
                    self.batch_size, self.seq_len, kv_head, self.head_size
                ),
            )
            self.assertEqual(
                value_compile,
                linear_outs[:, :, self.hidden_size + kv_head * self.head_size :].view(
                    self.batch_size, self.seq_len, kv_head, self.head_size
                ),
            )


@unittest.skipIf(
    skip_test_pt_2_3, "Skipping test as OP support available from PyTorch 2.3"
)
class MiniMHATester(TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        random.seed(SEED)
        self.mha = MaskedMHATest()
        self.beam_size_list = [1]
        self.batch_size_list = [1]
        self.head_size = 256
        self.head_num = 16
        self.head_num_kv_list = [1]
        self.max_seq_len = 64
        self.first_seq_len = 32

    def tearDown(self):
        del self.mha

    def test_mha(self):
        self.mha._test_mha(
            self.beam_size_list,
            self.batch_size_list,
            self.head_size,
            self.head_num,
            self.head_num_kv_list,
            self.max_seq_len,
            self.first_seq_len,
        )


if __name__ == "__main__":
    print("Seed is", SEED)
    run_tests()
