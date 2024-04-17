# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import unittest
import copy
import torch.nn as nn
from importlib import metadata
from torch.fx.experimental.proxy_tensor import make_fx
from parameterized import parameterized
from itertools import product

try:
    import zentorch

    HAS_ZENTORCH = True
except ImportError:
    HAS_ZENTORCH = False

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


class Test_Data:

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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class Test_MM_OP(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_matmul_variants(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        # mm
        self.assertEqual(
            torch._C._VariableFunctions.mm(data.x, data.y),
            torch.ops.zentorch.zendnn_mm(data.x, data.y),
        )
        self.assertEqual(
            torch.matmul(data.x, data.y), torch.ops.zentorch.zendnn_mm(data.x, data.y)
        )
        self.assertEqual(
            torch.mm(data.x, data.y), torch.ops.zentorch.zendnn_mm(data.x, data.y)
        )

        self.assertEqual(data.x @ data.y, torch.ops.zentorch.zendnn_mm(data.x, data.y))

        self.assertEqual(
            torch.mul(data.A, data.B), torch.ops.zentorch.zendnn_mm(data.A, data.B)
        )

    @parameterized.expand(supported_dtypes)
    def test_mm_mismatched_dimensions(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_mm(
                data.x,
                torch.reshape(
                    data.x, (1, list(data.x.shape)[0], list(data.x.shape)[1])
                ),
            )
        self.assertTrue(
            "zendnn_mm:  unsupported dims for self and mat2" in str(context.exception)
        )

    @parameterized.expand([("int",)])
    def test_mm_unsupported_dtype(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_mm(data.x, data.y)
        self.assertTrue(
            "zendnn_matmul: zendnn_matmul only supports Float and BFloat16"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_mm_relu(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        # mm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.mm(data.x, data.y)
            ),
            torch.ops.zentorch.zendnn_mm(data.x, data.y, fuse=1),
        )


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class Test_ADDMM_OP(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_addmm_variants(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        # addmm
        self.assertEqual(
            torch._C._VariableFunctions.addmm(data.input, data.x, data.y),
            torch.ops.zentorch.zendnn_addmm(
                data.input,
                data.x,
                data.y,
            ),
        )
        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(data.input, data.x, data.y, beta=1.3),
            torch.ops.zentorch.zendnn_addmm(data.input, data.x, data.y, beta=1.3),
        )

        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(data.input, data.x, data.y, alpha=1.3),
            torch.ops.zentorch.zendnn_addmm(data.input, data.x, data.y, alpha=1.3),
        )

        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                data.input, data.x, data.y, alpha=1.3, beta=1.3
            ),
            torch.ops.zentorch.zendnn_addmm(
                data.input, data.x, data.y, alpha=1.3, beta=1.3
            ),
        )

        # addmm with 1-d input
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zendnn_addmm(
                    data.input1d, data.x, data.y, alpha=1.3, beta=1.3
                )
                self.assertTrue(
                    "zendnn_matmul: zendnn_matmul is not supported for "
                    "bf16 tensors when bias is defined and alpha is not equal "
                    "to 1" in str(context.exception)
                )
        else:
            self.assertEqual(
                torch._C._VariableFunctions.addmm(
                    data.input1d, data.x, data.y, alpha=1.3, beta=1.3
                ),
                torch.ops.zentorch.zendnn_addmm(
                    data.input1d, data.x, data.y, alpha=1.3, beta=1.3
                ),
            )

    @parameterized.expand(supported_dtypes)
    def test_addmm_mismatched_dimensions(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_addmm(
                data.x,
                data.x,
                torch.reshape(
                    data.x, (list(data.x.shape)[0], list(data.x.shape)[1], 1)
                ),
            )

        self.assertTrue(
            "zendnn_addmm:  unsupported dims for self, mat1 and mat2"
            in str(context.exception)
        )

    @parameterized.expand(["int"])
    def test_addmm_unsupported_dtype(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_addmm(data.x, data.x, data.x)

        self.assertTrue(
            "zendnn_matmul: zendnn_matmul only supports Float and BFloat16"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_addmm_relu_with_kw(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        # addmm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    data.input, data.x, data.y, beta=1.5, alpha=1.7
                )
            ),
            torch.ops.zentorch.zendnn_addmm(
                data.input, data.x, data.y, beta=1.5, alpha=1.7, fuse=1
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(data.input, data.x, data.y, alpha=1.7)
            ),
            torch.ops.zentorch.zendnn_addmm(
                data.input, data.x, data.y, alpha=1.7, fuse=1
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(data.input, data.x, data.y, beta=1.5)
            ),
            torch.ops.zentorch.zendnn_addmm(
                data.input, data.x, data.y, beta=1.5, fuse=1
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(data.input, data.x, data.y, beta=0.0)
            ),
            torch.ops.zentorch.zendnn_addmm(
                data.input, data.x, data.y, beta=0.0, fuse=1
            ),
        )

    @parameterized.expand(supported_dtypes)
    def test_addmm_with_zero_alpha(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.addmm(data.input, data.x, data.y, alpha=0.0),
            torch.ops.zentorch.zendnn_addmm(data.input, data.x, data.y, alpha=0.0),
        )

    @parameterized.expand(supported_dtypes)
    def test_addmm_relu_without_kw(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        # addmm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(data.input, data.x, data.y)
            ),
            torch.ops.zentorch.zendnn_addmm(data.input, data.x, data.y, fuse=1),
        )


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class Test_BMM_OP(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_bmm_variants(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.bmm(data.x3d, data.y3d),
            torch.ops.zentorch.zendnn_bmm(data.x3d, data.y3d),
        )

    @parameterized.expand(supported_dtypes)
    def test_bmm_unsupported_dims(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_bmm(data.x, data.y)

        self.assertTrue(
            "zendnn_bmm:  unsupported dims for self and mat2" in str(context.exception)
        )

    @parameterized.expand([("int",)])
    def test_bmm_unsupported_dtype(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_bmm(data.x3d, data.y3d)

        self.assertTrue(
            "zendnn_matmul: zendnn_matmul only supports Float and BFloat16"
            in str(context.exception)
        )


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class Test_BADDBMM_OP(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_baddbmm_variants(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(data.input3d, data.x3d, data.y3d),
            torch.ops.zentorch.zendnn_baddbmm(data.input3d, data.x3d, data.y3d),
        )

    @parameterized.expand([("int",)])
    def test_baddbmm_unsupported_dtype(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_baddbmm(data.input3d, data.x3d, data.y3d)

        self.assertTrue(
            "zendnn_matmul: zendnn_matmul only supports Float and BFloat16"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_baddbmm_unsupported_dims(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_baddbmm(
                data.input3d.reshape((data.b * data.m), data.n), data.x3d, data.y3d
            )

        self.assertTrue(
            "zendnn_baddbmm:  unsupported dims for self, batch1 and batch2"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_baddbmm_with_kw(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                data.input3d, data.x3d, data.y3d, alpha=1.4
            ),
            torch.ops.zentorch.zendnn_baddbmm(
                data.input3d, data.x3d, data.y3d, alpha=1.4
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                data.input3d, data.x3d, data.y3d, beta=1.4
            ),
            torch.ops.zentorch.zendnn_baddbmm(
                data.input3d, data.x3d, data.y3d, beta=1.4
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                data.input3d, data.x3d, data.y3d, alpha=1.4, beta=1.3
            ),
            torch.ops.zentorch.zendnn_baddbmm(
                data.input3d, data.x3d, data.y3d, alpha=1.4, beta=1.3
            ),
        )


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class Test_MATMUL_IMPL_OP(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_zendnn_matmul_impl_for_mv_and_dot(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        # mv
        self.assertEqual(
            torch.mv(data.input, data.input1d),
            zentorch._C.zendnn_matmul_impl(
                data.input, data.input1d, data.empty_bias, data.result_m
            ),
        )

        # dot
        self.assertEqual(
            torch.dot(data.input1d, data.input1d),
            zentorch._C.zendnn_matmul_impl(
                data.input1d, data.input1d, data.empty_bias, data.result_1
            ),
        )


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TEST_EMBEDDING_BAG(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_embedding_bag_zendnn(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zendnn_embedding_bag(
                    data.embedding_matrix,
                    data.emb_input,
                    data.offsets,
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
                data.embedding_matrix,
                data.emb_input,
                data.offsets,
                False,
                0,
                False,
                None,
                False,
            )

            y_ebz, _, _, _ = torch.ops.zentorch.zendnn_embedding_bag(
                data.embedding_matrix,
                data.emb_input,
                data.offsets,
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
        data = Test_Data()
        data.create_data(dtype)

        # max mode is not supported whenever any of the sparse_opt
        # or scale_grad_opt is True
        y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(
            data.embedding_matrix,
            data.emb_input,
            data.offsets,
            scale_opt,
            mode,
            sprs_opt,
            None,
            include_last_offset,
        )
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zendnn_embedding_bag(
                    data.embedding_matrix,
                    data.emb_input,
                    data.offsets,
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
            y_ebz, _, _, _ = torch.ops.zentorch.zendnn_embedding_bag(
                data.embedding_matrix,
                data.emb_input,
                data.offsets,
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
        data = Test_Data()
        new_dtype = data.get_torch_type(dtype)
        model = CustomModelEmbeddingBagNN(100, 10, dtype=new_dtype)
        input = torch.randint(0, 10000, (1, 10))
        model_output = model(input)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(input)
        self.assertAlmostEqual(model_output.item(), compiled_graph_output.item())


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TEST_EMBEDDING(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_embedding_zendnn(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zendnn_embedding(
                    data.embedding_matrix, data.emb_input
                )
            self.assertTrue(
                "Only fp32 type weights are supported in ZenDNN Embedding!"
                in str(context.exception)
            )

        else:
            y_eb = torch._C._VariableFunctions.embedding(
                data.embedding_matrix, data.emb_input
            )

            y_ebz = torch.ops.zentorch.zendnn_embedding(
                data.embedding_matrix, data.emb_input
            )

            self.assertEqual(y_eb, y_ebz)

    @parameterized.expand(supported_dtypes)
    def test_embedding_sparse_scale(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        sparse_opt = [True, False]
        scale_grad_opt = [True, False]

        for sprs_opt in sparse_opt:
            for scale_opt in scale_grad_opt:
                if dtype == "bfloat16":
                    with self.assertRaises(RuntimeError) as context:
                        torch.ops.zentorch.zendnn_embedding(
                            data.embedding_matrix,
                            data.emb_input,
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
                        data.embedding_matrix, data.emb_input, -1, scale_opt, sprs_opt
                    )

                    y_ebz = torch.ops.zentorch.zendnn_embedding(
                        data.embedding_matrix, data.emb_input, -1, scale_opt, sprs_opt
                    )

                    self.assertEqual(y_eb, y_ebz)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_custom_embedding_compile(self, dtype):
        data = Test_Data()
        new_dtype = data.get_torch_type(dtype)
        model = CustomModelEmbeddingNN(100, dtype=new_dtype)
        input = torch.randint(0, 10000, (10,))
        model_output = model(input)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(input)
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TEST_EMBEDDING_BAG_GROUP(TestCase):

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
        data = Test_Data()
        data.create_data(dtype)
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zendnn_horizontal_embedding_bag_group(
                    [data.embedding_matrix] * 3,
                    [data.emb_input] * 3,
                    [data.offsets] * 3,
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
                data.embedding_matrix,
                data.emb_input,
                data.offsets,
                scale_opt,
                mode,
                sprs_opt,
                None,
                include_last_offset,
            )

            y_ebz_list = torch.ops.zentorch.zendnn_horizontal_embedding_bag_group(
                [data.embedding_matrix] * 3,
                [data.emb_input] * 3,
                [data.offsets] * 3,
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

        test_data = Test_Data()
        test_data.create_data(dtype)
        model = CustomModelEmbeddingBagGroup(test_data.R)
        indices = test_data.emb_input
        offsets = test_data.offsets

        fx_g = make_fx(model)(indices, offsets)
        fx_g_output = fx_g(indices, offsets)

        fx_g_optimized = zentorch.optimize(fx_g)

        fx_g_optimized_output = fx_g_optimized(indices, offsets)

        self.assertEqual(fx_g_output, fx_g_optimized_output)

        target = torch.ops.zentorch.zendnn_horizontal_embedding_bag_group.default
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

        test_data = Test_Data()
        test_data.create_data(dtype)
        model = CustomModelEmbeddingBagGroup(test_data.R)
        indices = test_data.emb_input
        offset = test_data.offsets

        native_output = model(indices, offset)
        torch._dynamo.reset()

        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices, offset)

        self.assertEqual(native_output, compiled_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TEST_EMBEDDING_GROUP(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_embedding_group_zendnn(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        if dtype == "bfloat16":
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zendnn_horizontal_embedding_group(
                    [data.embedding_matrix] * 3,
                    [data.emb_input] * 3,
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
                data.embedding_matrix, data.emb_input
            )

            y_ebz_list = torch.ops.zentorch.zendnn_horizontal_embedding_group(
                [data.embedding_matrix] * 3,
                [data.emb_input] * 3,
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

        test_data = Test_Data()
        test_data.create_data(dtype)
        model = CustomModelEmbeddingGroup(test_data.R)
        x = test_data.emb_input

        fx_g = make_fx(model)(x)
        fx_g_output = fx_g(x)

        fx_g_optimized = zentorch.optimize(fx_g)

        fx_g_optimized_output = fx_g_optimized(x)

        self.assertEqual(fx_g_output, fx_g_optimized_output)

        target = torch.ops.zentorch.zendnn_horizontal_embedding_group.default
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

        test_data = Test_Data()
        test_data.create_data(dtype)
        model = CustomModelEmbeddingGroup(test_data.R)
        x = test_data.emb_input

        native_output = model(x)

        torch._dynamo.reset()
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

        test_data = Test_Data()
        test_data.create_data(dtype)
        model = CustomModel_Emb_EmbBag_Common_Node(test_data.R)
        indices = test_data.emb_input
        offsets = test_data.offsets

        native_output = model(indices, offsets)
        torch._dynamo.reset()
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

        test_data = Test_Data()
        test_data.create_data(dtype)
        model = CustomModel_Emb_EmbBag_Diff_Node(test_data.R)
        indices = test_data.emb_input
        offsets = test_data.offsets

        native_output = model(indices, offsets)
        torch._dynamo.reset()
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

        test_data = Test_Data()
        test_data.create_data(dtype)
        model = CustomModel_2D_Embedding(test_data.R)
        indices = torch.cat([torch.unsqueeze(test_data.emb_input, dim=0)] * 2)

        native_output = model(indices)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices)
        self.assertEqual(native_output, compiled_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TEST_HORIZONTAL_MLP(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_horizontal_mlp(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = Custom_Horizontal_GroupMLP_Model(data.get_torch_type(dtype))

        for i in range(len(data.x2)):
            for j in range(len(data.y2)):
                native_output = model(data.x2[i], data.y2[j])
                torch._dynamo.reset()
                compiled_graph = torch.compile(model, backend="zentorch")

                compiled_output = compiled_graph(data.x2[i], data.y2[j])
                self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_horizontal_mlp_multi_user(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = Custom_Horizontal_GroupMLP_multi_user_Model(
            -1, data.get_torch_type(dtype)
        )
        for i in range(len(data.x2)):
            for j in range(len(data.y2)):
                native_output = model(data.x2[i], data.y2[j])
                torch._dynamo.reset()
                compiled_graph = torch.compile(model, backend="zentorch")

                compiled_output = compiled_graph(data.x2[i], data.y2[j])
                self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    def test_horizontal_mlp_unsupported_dims_1(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_attn_horizontal_mlp_group(
                [data.x1[0]] * 3,
                [data.y1[0]] * 3,
                [data.y1[0]] * 3,
                [0.0] * 3,
                [1.0] * 3,
                [0] * 3,
                [1],
            )
        self.assertTrue(
            "zendnn_addmm: unsupported dims for self, mat1 and mat2"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_horizontal_mlp_unsupported_dims_2(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_attn_horizontal_mlp_group(
                [data.y2[0]] * 3,
                [data.y2[0]] * 3,
                [data.y2[0]] * 3,
                [0.0] * 3,
                [1.0] * 3,
                [0] * 3,
                [1],
            )
        self.assertTrue(
            "zendnn_addmm:  unsupported dims for self, mat1 and mat2"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_horizontal_mlp_input_shape_compatibility(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_attn_horizontal_mlp_group(
                [data.M[1]] * 3,
                [data.y1[0]] * 3,
                [data.y1[0]] * 3,
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
        data = Test_Data()
        data.create_data("bfloat16")
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_addmm(data.input1d, data.x, data.y, alpha=1.7)
            self.assertTrue(
                "zendnn_matmul: zendnn_matmul is not supported for bf16 \
                tensors when bias is defined and alpha is not equal to 1"
                in str(context.exception)
            )


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TEST_GROUP_MLP(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_mlp_model(self, dtype):
        data = Test_Data()
        data.create_data(dtype)

        model = CustomModel_GroupMLP_Model(data.k, data.get_torch_type(dtype))

        native_output = model(data.x)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(data.x)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_mlp_model_relu(self, dtype):
        data = Test_Data()
        data.create_data(dtype)

        model = CustomModel_GroupMLP_Model_Relu(data.k, data.get_torch_type(dtype))

        native_output = model(data.x)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(data.x)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_mlp_model_relu_gelu(self, dtype):
        data = Test_Data()
        data.create_data(dtype)

        model = CustomModel_GroupMLP_Model_Relu_Gelu(data.k, data.get_torch_type(dtype))

        native_output = model(data.x)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(data.x)
        self.assertEqual(native_output, compiled_output, atol=1e-3, rtol=1e-5)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_incorrect_dims(self, dtype):
        data = Test_Data()
        data.create_data(dtype)

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_addmm(data.x3d, data.x, data.x)
            self.assertTrue(
                "zendnn_addmm: unsupported dims for self, mat1 and mat2!"
                in str(context.exception)
            )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_addmm_1dbias(data.x, data.x, data.x)
            self.assertTrue(
                "zendnn_addmm_1dbias: unsupported dims for self, mat1 and mat2!"
                in str(context.exception)
            )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_baddbmm(data.x, data.x3d, data.x3d)
            self.assertTrue(
                "zendnn_baddbmm:  unsupported dims for self, batch1 and batch2!"
                in str(context.exception)
            )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_mm(data.x3d, data.x3d)
            self.assertTrue(
                "zendnn_mm:  unsupported dims for self and mat2!"
                in str(context.exception)
            )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_bmm(data.x, data.x)
            self.assertTrue(
                "zendnn_bmm:  unsupported dims for self and mat2!"
                in str(context.exception)
            )

    @torch.inference_mode()
    def test_bf16_alpha_not_1(self):
        data = Test_Data()
        data.create_data("bfloat16")

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_addmm(data.input1d, data.x, data.y, alpha=1.7)
            self.assertTrue(
                "zendnn_matmul: zendnn_matmul is not supported for bf16 \
                tensors when bias is defined and alpha is not equal to 1"
                in str(context.exception)
            )


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TEST_GROUP_EB_MLP(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_eb_mlp_model(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN EmbeddingBag!"
            )

        data = Test_Data()
        data.create_data(dtype)

        indices = data.emb_input
        offsets = data.offsets

        model = CustomModel_Group_EB_MLP_Model(data.R, data.k)

        native_output = model(indices, offsets, data.x)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices, offsets, data.x)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_mlp_eb_model(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN EmbeddingBag!"
            )

        data = Test_Data()
        data.create_data(dtype)

        indices = data.emb_input
        offsets = data.offsets

        model = CustomModel_Group_MLP_EB_Model(data.R, data.k)

        native_output = model(indices, offsets, data.x)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices, offsets, data.x)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_eb_mlp_model_multiple_groups(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN EmbeddingBag!"
            )

        eb_inputs = torch.randint(0, 3, (2, 1))
        mlp_inputs = torch.randn(2, 2)

        model = CustomModel_Group_EB_MLP_Model_multiple_groups()
        native_output = model(eb_inputs, mlp_inputs)
        torch._dynamo.reset()
        compiled_model = torch.compile(model, backend="zentorch")
        compiled_output = compiled_model(eb_inputs, mlp_inputs)

        self.assertEqual(native_output, compiled_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
# Testing revealed one of the corner cases where the common output node can
# have heterogeneous nodes like embedding1, embedding2, sum1, sum2, embedding3.
# To test the above scenario, the following testcases are added.
# Both the group ops are being tested here, with the heterogeneous op being sum
class TEST_GROUP_EMBED_OPS_WITH_SUM_OPS(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_group_eb_with_sum(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Skipping test case since only fp32 type weights are supported \
                in ZenDNN Embedding!"
            )

        data = Test_Data()
        data.create_data(dtype)

        indices = data.emb_input
        offsets = data.offsets

        model = CustomModel_EmbeddingBag_Sum_nodes(data.R)

        native_output = model(indices, offsets)
        torch._dynamo.reset()
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

        data = Test_Data()
        data.create_data(dtype)

        indices = data.emb_input

        model = CustomModel_Embedding_Sum_nodes(data.R)

        native_output = model(indices)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_output = compiled_graph(indices)
        self.assertEqual(native_output, compiled_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TestBF16Device(TestCase):
    @unittest.skipIf(
        not zentorch._C.is_bf16_supported(), "CPU does not support AVX512 BF16."
    )
    def test_bf16_device(self):
        self.assertTrue(zentorch._C.is_bf16_supported(), "CPU supports AVX512 BF16.")


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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
    def __init__(self):
        super(CustomModel_Group_EB_MLP_Model_multiple_groups, self).__init__()
        # Common Nodes
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.eb_bags = [torch.nn.EmbeddingBag(3, 3, mode="sum") for _ in range(2)]

        self.bmlp_0 = torch.nn.Linear(2, 4)
        self.bmlp_1 = torch.nn.Linear(4, 4)
        self.bmlp_2 = torch.nn.Linear(4, 3)

        self.tmlp_0 = torch.nn.Linear(12, 4)
        self.tmlp_1 = torch.nn.Linear(4, 2)
        self.tmlp_2 = torch.nn.Linear(2, 2)
        self.tmlp_3 = torch.nn.Linear(2, 1)

    def forward(self, eb_inputs, mlp_inputs):

        outputs = []

        for _ in range(3):
            eb_outputs = [eb_op(eb_inputs, None) for eb_op in self.eb_bags]

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

        return torch.sum(torch.concat(outputs, dim=1))


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "PT PLUGIN is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class CustomModelEmbeddingNN(nn.Module):
    def __init__(self, embedding_dim, dtype=torch.float):
        super(CustomModelEmbeddingNN, self).__init__()
        self.embedding = nn.Embedding(10000, embedding_dim, dtype=dtype)

    def forward(self, input):
        embed = self.embedding(input)
        return embed


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class CustomModel_2D_Embedding(nn.Module):
    def __init__(self, num_embeddings):
        super(CustomModel_2D_Embedding, self).__init__()
        self.embedding_1 = torch.nn.Embedding(num_embeddings, 3)
        self.embedding_2 = torch.nn.Embedding(num_embeddings, 3)

    def forward(self, inputs):
        output = self.embedding_1(inputs) + self.embedding_2(inputs)

        return output


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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
        eb_sum_0 = torch.unsqueeze(torch.sum(torch.cat(eb_outputs_grp_0), dim=0), dim=0)

        eb_outputs_grp_1 = [
            self.eb_bags_grp_1[i](eb_input, eb_offset) for i in range(10)
        ]
        eb_sum_1 = torch.unsqueeze(torch.sum(torch.cat(eb_outputs_grp_1), dim=0), dim=0)

        eb_outputs_grp_2 = [
            self.eb_bags_grp_2[i](eb_input, eb_offset) for i in range(6)
        ]
        eb_sum_2 = torch.unsqueeze(torch.sum(torch.cat(eb_outputs_grp_2), dim=0), dim=0)

        output = torch.cat([eb_sum_0, eb_sum_1, eb_sum_2])

        return output


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
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


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TestMMRELU(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_mm_relu_optimize(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelMMRelu2().eval()
        for i in range(len(data.x1)):
            for j in range(len(data.y1)):
                model_output = model(data.x1[i], data.y1[j])
                torch._dynamo.reset()
                compiled_graph = torch.compile(model, backend="zentorch")
                compiled_graph_output = compiled_graph(data.x1[i], data.y1[j])
                self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zero_input_optimize(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelMMRelu2().eval()
        model_output = model(data.x1[0] * 0, data.y1[0] * 0)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(data.x1[0] * 0, data.y1[0] * 0)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_negative_input_optimize(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelMMRelu2().eval()
        model_output = model(data.x1[0] * -1, data.y1[0] * -1)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(data.x1[0] * -1, data.y1[0] * -1)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_custom_mm_relu1(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelMMReLU1(data.n, data.n - 2, data.n - 5).eval()
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(data.input)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(data.input)
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TestMMADD(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_mm_add_optimize(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelMMAdd1().eval()
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")
        for inp in data.M:
            for i in range(len(data.x1)):
                for j in range(len(data.y1)):
                    torch._dynamo.reset()
                    zentorch_model = copy.deepcopy(model)
                    inductor_graph = torch.compile(model, backend="inductor")
                    inductor_graph_output = inductor_graph(inp, data.x1[i], data.y1[j])
                    torch._dynamo.reset()
                    zentorch_graph = torch.compile(zentorch_model, backend="zentorch")
                    zentorch_graph_output = zentorch_graph(inp, data.x1[i], data.y1[j])
                    self.assertEqual(inductor_graph_output, zentorch_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zero_input(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelMMAdd1().eval()
        for inp in data.M:
            model_output = model(inp * 0, data.x1[0] * 0, data.y1[0] * 0)
            torch._dynamo.reset()
            compiled_graph = torch.compile(model, backend="zentorch")
            compiled_graph_output = compiled_graph(
                inp * 0, data.x1[0] * 0, data.y1[0] * 0
            )
            self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_inf_input(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelMMAdd1().eval()
        for inp in data.M:
            model_output = model(inp / 0, data.x1[0] / 0, data.y1[0] / 0)
            torch._dynamo.reset()
            compiled_graph = torch.compile(model, backend="zentorch")
            compiled_graph_output = compiled_graph(
                inp / 0, data.x1[0] / 0, data.y1[0] / 0
            )
            self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_nan_input(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelMMAdd1().eval()
        for inp in data.M:
            model_output = model(
                inp * float("nan"), data.x1[0] * float("nan"), data.y1[0] * float("nan")
            )
            torch._dynamo.reset()
            compiled_graph = torch.compile(model, backend="zentorch")
            compiled_graph_output = compiled_graph(
                inp * float("nan"), data.x1[0] * float("nan"), data.y1[0] * float("nan")
            )
            self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_identity_input_nan(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        if dtype == "bfloat16":
            self.skipTest("Skipping it since this testcase is not applicable for BF16.")
        model = CustomModelMMAdd1().eval()
        model_output = model(
            torch.eye(data.M[0].shape[0], data.M[0].shape[1]),
            data.x1[0] * float("nan"),
            data.y1[0] * float("nan"),
        )
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(
            torch.eye(data.M[0].shape[0], data.M[0].shape[1]),
            data.x1[0] * float("nan"),
            data.y1[0] * float("nan"),
        )
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TestADDMM_GELU(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_addmm_gelu_tanh(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelAddmmGeluTanh().eval()
        for inp in data.M:
            for i in range(len(data.x1)):
                for j in range(len(data.y1)):
                    model_output = model(inp, data.x1[i], data.y1[j])
                    torch._dynamo.reset()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(inp, data.x1[i], data.y1[j])
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_addmm_gelu_exact(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelAddmmGeluExact().eval()
        for inp in data.M:
            for i in range(len(data.x1)):
                for j in range(len(data.y1)):
                    model_output = model(inp, data.x1[i], data.y1[j])
                    torch._dynamo.reset()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(inp, data.x1[i], data.y1[j])
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_addmm_gelu(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelAddmmGelu().eval()
        for inp in data.M:
            for i in range(len(data.x1)):
                for j in range(len(data.y1)):
                    model_output = model(inp, data.x1[i], data.y1[j])
                    torch._dynamo.reset()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(inp, data.x1[i], data.y1[j])
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_addmm_view_gelu(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelMM_View_Unary_OP().eval()
        for inp in data.M:
            for i in range(len(data.x1)):
                for j in range(len(data.y1)):
                    model_output = model(inp, data.x1[i], data.y1[j])
                    torch._dynamo.reset()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(inp, data.x1[i], data.y1[j])
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_mm_diff_user_in_btw(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelMM_Diff_User_In_Btw().eval()
        for inp in data.M:
            for i in range(len(data.x1)):
                for j in range(len(data.y1)):
                    model_output = model(inp, data.x1[i], data.y1[j])
                    torch._dynamo.reset()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(inp, data.x1[i], data.y1[j])
                    self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TestADDMM_RELU(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_addmm_relu(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue BF16 path.")
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelAddmmRelu2().eval()
        for inp in data.M:
            for i in range(len(data.x1)):
                for j in range(len(data.y1)):
                    model_output = model(inp, data.x1[i], data.y1[j])
                    torch._dynamo.reset()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = compiled_graph(inp, data.x1[i], data.y1[j])
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_custom_addmm_relu1(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelAddmmReLU1(data.n, data.n - 2).eval()
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(data.input)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(data.input)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    @unittest.skip("Nan and Inf giving non-deterministic output")
    def test_custom_addmm_relu1_with_nan_or_inf(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it since this testcase is not applicable for BF16.")
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelAddmmReLU1(data.n, data.n - 2).eval()
        # Nan's output is non-deterministic. Skipping Nan
        # data.input[0][0] = float("nan")
        data.input[1][1] = float("inf")
        torch._dynamo.reset()
        zentorch_model = copy.deepcopy(model)
        inductor_graph = torch.compile(model, backend="inductor")
        inductor_graph_output = inductor_graph(data.input)
        torch._dynamo.reset()
        zentorch_graph = torch.compile(zentorch_model, backend="zentorch")
        zentorch_graph_output = zentorch_graph(data.input)
        self.assertEqual(inductor_graph_output, zentorch_graph_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TestLinear_Relu(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_linear_relu(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = nn.Sequential(nn.Linear(data.n, data.m), nn.ReLU())
        if dtype == "bfloat16":
            model = model.bfloat16()
        fx_g = make_fx(model)(data.input)
        fx_g_modified = zentorch.optimize(fx_g)
        fx_g_output = fx_g(data.input)
        fx_g_modified_output = fx_g_modified(data.input)
        self.assertEqual(fx_g_output, fx_g_modified_output)
        for node in fx_g_modified.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                if node.target.name() in ["aten::addmm"]:
                    self.assertEqual(
                        node.target, torch.ops.zentorch.zendnn_addmm_1dbias
                    )


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TestLinear_Gelu(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_linear_gelu_tanh(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = nn.Sequential(nn.Linear(data.n, data.m), nn.GELU(approximate="tanh"))
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(data.input)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(data.input)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_linear_gelu_none(self, dtype):
        data = Test_Data()
        data.create_data(dtype)
        model = nn.Sequential(nn.Linear(data.n, data.m), nn.GELU(approximate="none"))
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(data.input)
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(data.input)
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not HAS_ZENTORCH, "ZENTORCH is not installed")
class TestBMMADD(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_bmm_baddbmm(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue with BF16 path.")
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelBMMAdd1().eval()
        for i in range(len(data.x2)):
            for j in range(len(data.y2)):
                model_output = model(data.M2, data.x2[i], data.y2[j])
                torch._dynamo.reset()
                compiled_graph = torch.compile(model, backend="zentorch")
                compiled_graph_output = compiled_graph(data.M2, data.x2[i], data.y2[j])
                self.assertEqual(
                    model_output, compiled_graph_output, atol=1e-5, rtol=1e-3
                )

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_zendnn_baddbmm_unsupport(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue with BF16 path.")
        data = Test_Data()
        data.create_data(dtype)
        model = CustomModelBMM_Unsupport().eval()
        model_output = model(data.M3, data.x2[0], data.y2[0])
        torch._dynamo.reset()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(data.M3, data.x2[0], data.y2[0])
        self.assertEqual(
            model_output, compiled_graph_output, atol=1e-5, rtol=1e-3
        )


if __name__ == "__main__":
    run_tests()
