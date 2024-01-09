# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import unittest
import torch.nn as nn
from importlib import metadata
from torch.fx.experimental.proxy_tensor import make_fx
from parameterized import parameterized

try:
    import torch_zendnn_plugin as zentorch
    HAS_PT_PLUGIN = True
except ImportError:
    HAS_PT_PLUGIN = False

supported_dtypes = [('fp32')]
if zentorch._C.is_bf16_supported():
    supported_dtypes.append(('bfloat16'))
else:
    print("Warning: Skipping Bfloat16 Testcases since they \
are not supported on this hardware")


class Test_Data:
    def __init__(self, dtype):
        self.dtypes = {'fp32': torch.float32, 'bfloat16': torch.bfloat16,
                       'int': torch.int}
        self.b = torch.randint(1, 11, (1,)).item()
        self.m = torch.randint(1, 11, (1,)).item()
        self.k = torch.randint(1, 11, (1,)).item()
        self.n = torch.randint(1, 11, (1,)).item()

        # m*k, k*n, m*n
        self.x = torch.randn(self.m, self.k).type(self.dtypes[dtype])
        self.y = torch.randn(self.k, self.n).type(self.dtypes[dtype])

        self.input = torch.randn(self.m, self.n).type(self.dtypes[dtype])

        self.A = torch.randn(self.m, 1).type(self.dtypes[dtype])
        self.B = torch.randn(1, self.m).type(self.dtypes[dtype])

        # b*m*k, b*k*n, b*m*n
        self.x3d = torch.randn(self.b, self.m, self.k).type(self.dtypes[dtype])
        self.y3d = torch.randn(self.b, self.k, self.n).type(self.dtypes[dtype])
        self.input3d = torch.randn(self.b, self.m, self.n).type(self.dtypes[dtype])

        self.R = torch.randint(11, 20, (1,)).item()
        self.W = torch.randint(1, 15, (1,)).item()
        self.embedding_matrix = torch.rand(self.R, 3).type(self.dtypes[dtype])
        self.emb_input = torch.randint(0, self.R, (self.W,))
        self.offsets = torch.tensor([0, self.W]).type(self.dtypes[dtype])

        self.M = torch.randn(60, 30).type(self.dtypes[dtype])

        self.x1 = [
            torch.randn(60, 40).type(self.dtypes[dtype]),
            torch.randn(40, 60).transpose(0, 1).type(self.dtypes[dtype]),
        ]

        self.y1 = [
            torch.randn(40, 30).type(self.dtypes[dtype]),
            torch.randn(30, 40).transpose(1, 0).type(self.dtypes[dtype]),
        ]

        self.M2 = torch.randn(60, 30, 50).type(self.dtypes[dtype])

        self.x2 = [
            torch.randn(60, 30, 40).type(self.dtypes[dtype]),
            torch.randn(60, 40, 30).transpose(1, 2).type(self.dtypes[dtype]),
            torch.randn(30, 60, 40).transpose(0, 1).type(self.dtypes[dtype]),
        ]

        self.y2 = [
            torch.randn(60, 40, 50).type(self.dtypes[dtype]),
            torch.randn(60, 50, 40).transpose(1, 2).type(self.dtypes[dtype]),
            torch.randn(50, 40, 60).transpose(0, 2).type(self.dtypes[dtype]),
        ]


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class Test_MM_OP(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_matmul_variants(self, dtype):
        data = Test_Data(dtype)
        # mm
        self.assertEqual(
            torch._C._VariableFunctions.mm(data.x, data.y),
            torch.ops.zentorch.zendnn_mm(data.x, data.y)
        )
        self.assertEqual(
            torch.matmul(data.x, data.y), torch.ops.zentorch.zendnn_mm(data.x, data.y)
        )
        self.assertEqual(
            torch.mm(data.x, data.y), torch.ops.zentorch.zendnn_mm(data.x, data.y)
        )

        self.assertEqual(
            data.x @ data.y, torch.ops.zentorch.zendnn_mm(data.x, data.y)
        )

        self.assertEqual(
            torch.mul(data.A, data.B), torch.ops.zentorch.zendnn_mm(data.A, data.B)
        )

    @parameterized.expand(supported_dtypes)
    def test_mm_mismatched_dimensions(self, dtype):
        data = Test_Data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_mm(data.x, torch.reshape(
                data.x, (1, list(data.x.shape)[0], list(data.x.shape)[1])))
        self.assertTrue("zendnn_mm:  unsupported dims for self and mat2"
                        in str(context.exception))

    @parameterized.expand([
        ('int',)
    ])
    def test_mm_unsupported_dtype(self, dtype):
        data = Test_Data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_mm(data.x, data.y)
        self.assertTrue("zendnn_mm: zendnn_mm only supports Float and BFloat16"
                        in str(context.exception))

    @parameterized.expand(supported_dtypes)
    def test_mm_relu(self, dtype):
        data = Test_Data(dtype)
        # mm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.mm(data.x, data.y)),
            torch.ops.zentorch.zendnn_mm(data.x, data.y, fuse=1)
        )


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class Test_ADDMM_OP(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_addmm_variants(self, dtype):
        data = Test_Data(dtype)
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
            torch._C._VariableFunctions.addmm(data.input, data.x,
                                              data.y, alpha=1.3, beta=1.3),
            torch.ops.zentorch.zendnn_addmm(data.input, data.x,
                                            data.y, alpha=1.3, beta=1.3)
        )

    @parameterized.expand(supported_dtypes)
    def test_addmm_mismatched_dimensions(self, dtype):
        data = Test_Data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_addmm(
                data.x, data.x, torch.reshape(data.x, (
                    list(data.x.shape)[0], list(data.x.shape)[1], 1)))

        self.assertTrue("zendnn_addmm:  unsupported dims for self, mat1 and mat2"
                        in str(context.exception))

    @parameterized.expand([
        ('int',)
    ])
    def test_addmm_unsupported_dtype(self, dtype):
        data = Test_Data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_addmm(data.x, data.x, data.x)

        self.assertTrue(
            "zendnn_addmm: zendnn_addmm only supports Float and BFloat16"
            in str(context.exception))

    @parameterized.expand(supported_dtypes)
    def test_addmm_relu_with_kw(self, dtype):
        data = Test_Data(dtype)
        # addmm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    data.input, data.x, data.y, beta=1.5, alpha=1.7)
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
        data = Test_Data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_addmm(data.input, data.x, data.y, alpha=0.0)

        self.assertTrue("zendnn_matmul: alpha is set to zero" in str(context.exception))

    @parameterized.expand(supported_dtypes)
    def test_addmm_relu_without_kw(self, dtype):
        data = Test_Data(dtype)
        # addmm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(data.input, data.x, data.y)
            ),
            torch.ops.zentorch.zendnn_addmm(
                data.input, data.x, data.y, fuse=1
            ),
        )


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class Test_BMM_OP(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_bmm_variants(self, dtype):
        data = Test_Data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.bmm(data.x3d, data.y3d),
            torch.ops.zentorch.zendnn_bmm(data.x3d, data.y3d),
        )

    @parameterized.expand(supported_dtypes)
    def test_bmm_unsupported_dims(self, dtype):
        data = Test_Data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_bmm(data.x, data.y)

        self.assertTrue(
            "zendnn_bmm:  unsupported dims for self and mat2"
            in str(context.exception))

    @parameterized.expand([
        ('int',)
    ])
    def test_bmm_unsupported_dtype(self, dtype):
        data = Test_Data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_bmm(data.x3d, data.y3d)

        self.assertTrue(
            "zendnn_bmm: zendnn_bmm only supports Float and BFloat16"
            in str(context.exception))


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class Test_BADDBMM_OP(TestCase):
    @parameterized.expand(supported_dtypes)
    def test_baddbmm_variants(self, dtype):
        data = Test_Data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(data.input3d, data.x3d, data.y3d),
            torch.ops.zentorch.zendnn_baddbmm(data.input3d, data.x3d, data.y3d),
        )

    @parameterized.expand([
        ('int',)
    ])
    def test_baddbmm_unsupported_dtype(self, dtype):
        data = Test_Data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_baddbmm(data.input3d, data.x3d, data.y3d)

        self.assertTrue(
            "zendnn_baddbmm: zendnn_baddbmm only supports Float and BFloat16"
            in str(context.exception))

    @parameterized.expand(supported_dtypes)
    def test_baddbmm_unsupported_dims(self, dtype):
        data = Test_Data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zendnn_baddbmm(
                data.input3d.reshape((data.b * data.m), data.n), data.x3d, data.y3d)

        self.assertTrue(
            "zendnn_baddbmm:  unsupported dims for self, batch1 and batch2"
            in str(context.exception))

    @parameterized.expand(supported_dtypes)
    def test_baddbmm_with_kw(self, dtype):
        data = Test_Data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                data.input3d, data.x3d, data.y3d, alpha=1.4),
            torch.ops.zentorch.zendnn_baddbmm(
                data.input3d, data.x3d, data.y3d, alpha=1.4),
        )

        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                data.input3d, data.x3d, data.y3d, beta=1.4),
            torch.ops.zentorch.zendnn_baddbmm(
                data.input3d, data.x3d, data.y3d, beta=1.4),
        )

        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                data.input3d, data.x3d, data.y3d, alpha=1.4, beta=1.3),
            torch.ops.zentorch.zendnn_baddbmm(
                data.input3d, data.x3d, data.y3d, alpha=1.4, beta=1.3),
        )


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TEST_EMBEDDING_BAG(Test_Data):
    @parameterized.expand(supported_dtypes)
    def test_embedding_bag_zendnn(self, dtype):
        data = Test_Data(dtype)

        y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(
            data.embedding_matrix, data.emb_input,
            data.offsets, False, 0, False, None, False
        )

        y_ebz, _, _, _ = torch.ops.zentorch.zendnn_embedding_bag(
            data.embedding_matrix, data.emb_input,
            data.offsets, False, 0, False, None, False, -1
        )

        self.assertEqual(y_eb, y_ebz)

    @parameterized.expand(supported_dtypes)
    def test_embedding_bag_sparse_scale_mode(self, dtype):
        data = Test_Data(dtype)

        # Issue with embedding bag with different modes
        sparse_opt = [True, False]
        scale_grad_opt = [True, False]

        i = 0
        while (i <= 0):
            for sprs_opt in sparse_opt:
                for scale_opt in scale_grad_opt:
                    y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(
                        data.embedding_matrix, data.emb_input, data.offsets, scale_opt,
                        i, sprs_opt, None, False)

                    y_ebz, _, _, _ = torch.ops.zentorch.zendnn_embedding_bag(
                        data.embedding_matrix, data.emb_input, data.offsets, scale_opt,
                        i, sprs_opt, None, False, -1)

                    self.assertEqual(y_eb, y_ebz)
            i = i + 1

    @torch.no_grad()
    def test_custom_embedding_bag(self):
        model = CustomModelEmbeddingBagNN(100, 10)
        input = torch.randint(0, 10000, (1, 10))
        model_output = model(input)
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(input)
        self.assertAlmostEqual(model_output.item(), compiled_graph_output.item())


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TEST_EMBEDDING_BAG_GROUP(Test_Data):
    @parameterized.expand(supported_dtypes)
    def test_embedding_bag_group_zendnn(self, dtype):
        data = Test_Data(dtype)

        y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(
            data.embedding_matrix, data.emb_input, data.offsets,
            False, 0, False, None, False
        )

        y_ebz_list = torch.ops.zentorch.zendnn_custom_embedding_bag_group(
            [data.embedding_matrix] * 3, [data.emb_input] * 3,
            [data.offsets] * 3, [False] * 3, [0] * 3, [False] * 3,
            [None] * 3, [False] * 3, [-1] * 3
        )

        for i in range(0, int(len(y_ebz_list) / 4)):
            self.assertEqual(y_eb, y_ebz_list[i * 4])

    @torch.no_grad()
    def test_group_embeddingbag(self):
        model = CustomModelEmbeddingBagGroup()
        x = {
            "eb_bags": {"input": torch.randint(0, 4, (5, 14)), "offset": None},
        }

        fx_g = make_fx(model)(
            x["eb_bags"]["input"],
            x["eb_bags"]["offset"],
        )
        fx_g_output = fx_g(
            x["eb_bags"]["input"],
            x["eb_bags"]["offset"],
        )

        fx_g_optimized = zentorch.optimize(fx_g)
        fx_g_optimized = zentorch._optimize.replace_emb_bag(fx_g_optimized)
        fx_g_optimized_output = fx_g_optimized(
            x["eb_bags"]["input"],
            x["eb_bags"]["offset"],
        )
        self.assertAlmostEqual(fx_g_output.item(), fx_g_optimized_output.item())

        target = torch.ops.zentorch.zendnn_custom_embedding_bag_group
        group_eb_count = 0

        for node in fx_g_optimized.graph.nodes:
            if (
                isinstance(node.target, torch._ops.OpOverloadPacket)
                and node.target == target
            ):
                group_eb_count += 1

        self.assertEqual(group_eb_count, 3)


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TestBF16Device(TestCase):
    @unittest.skipIf(
        not zentorch._C.is_bf16_supported(), "CPU does not support AVX512 BF16."
    )
    def test_bf16_device(self):
        self.assertTrue(zentorch._C.is_bf16_supported(), "CPU supports AVX512 BF16.")


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class CustomModelEmbeddingBagNN(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(CustomModelEmbeddingBagNN, self).__init__()
        self.embedding = nn.EmbeddingBag(10000, embedding_dim)
        self.intermediate = nn.Linear(embedding_dim, output_dim)
        self.output = nn.Linear(output_dim, 1)

    def forward(self, input):
        embed = self.embedding(input)
        intermediate = self.intermediate(embed)
        output = self.output(intermediate)
        return output


class CustomModelEmbeddingBagGroup(nn.Module):
    def __init__(self):
        super(CustomModelEmbeddingBagGroup, self).__init__()
        self.eb_bags_grp_0 = [
            torch.nn.EmbeddingBag(5, 14, mode="sum")
        ] * 5
        self.eb_bags_grp_1 = [
            torch.nn.EmbeddingBag(5, 14, mode="sum")
        ] * 10
        self.eb_bags_grp_2 = [
            torch.nn.EmbeddingBag(5, 14, mode="sum")
        ] * 6

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

        output = torch.sum(torch.cat([eb_sum_0, eb_sum_1, eb_sum_2]))

        return output


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TestZenTorchVersion(TestCase):
    def test_plugin_version(self):
        self.assertTrue(zentorch.__version__, metadata.version("torch-zendnn-plugin"))


class CustomModelBMMAdd1(nn.Module):
    def __init__(self):
        super(CustomModelBMMAdd1, self).__init__()

    def forward(self, input, batch1, batch2):
        bmm_res = torch.bmm(batch1, batch2)
        add_res = torch.add(bmm_res, input)
        baddbmm_res = torch.baddbmm(add_res, batch1, batch2, beta=1.5, alpha=1.4)
        return baddbmm_res


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


class CustomModelAddmmGelu2(nn.Module):
    def __init__(self):
        super(CustomModelAddmmGelu2, self).__init__()
        self.gelu = nn.GELU(approximate="tanh")
        self.gelu2 = nn.GELU()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        GELU1_res = self.gelu(add_res)
        addmm_res = torch.addmm(GELU1_res, batch1, batch2)
        GELU2_res = self.gelu2(addmm_res)
        return GELU2_res


class CustomModelAddmmGelu1(nn.Module):
    def __init__(self):
        super(CustomModelAddmmGelu1, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        GELU1_res = nn.functional.gelu(add_res, approximate="tanh")
        addmm_res = torch.addmm(GELU1_res, batch1, batch2, beta=1.7, alpha=1.6)
        GELU2_res = torch._C._nn.gelu_(addmm_res, approximate="none")
        return GELU2_res


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TestMMRELU(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_mm_relu_optimize(self, dtype):
        data = Test_Data(dtype)
        model = CustomModelMMRelu2().eval()
        for i in range(len(data.x1)):
            for j in range(len(data.y1)):
                model_output = model(data.x1[i], data.y1[j])
                compiled_graph = torch.compile(model, backend='zentorch')
                compiled_graph_output = compiled_graph(data.x1[i], data.y1[j])
                self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_zero_input_optimize(self, dtype):
        data = Test_Data(dtype)
        model = CustomModelMMRelu2().eval()
        model_output = model(data.x1[0] * 0, data.y1[0] * 0)
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(data.x1[0] * 0, data.y1[0] * 0)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_negative_input_optimize(self, dtype):
        data = Test_Data(dtype)
        model = CustomModelMMRelu2().eval()
        model_output = model(data.x1[0] * -1, data.y1[0] * -1)
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(data.x1[0] * -1, data.y1[0] * -1)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_custom_mm_relu1(self, dtype):
        data = Test_Data(dtype)
        model = CustomModelMMReLU1(data.n, data.n - 2, data.n - 5).eval()
        if dtype == 'bfloat16':
            model = model.bfloat16()
        model_output = model(data.input)
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(data.input)
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TestMMADD(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_mm_add_optimize(self, dtype):
        data = Test_Data(dtype)
        model = CustomModelMMAdd1().eval()
        if dtype == 'bfloat16':
            self.skipTest("Skipping it due to issue BF16 path.")
            model = model.bfloat16()
        for i in range(len(data.x1)):
            for j in range(len(data.y1)):
                inductor_graph = torch.compile(model, backend='inductor')
                inductor_graph_output = inductor_graph(data.M, data.x1[i], data.y1[j])
                compiled_graph = torch.compile(model, backend='zentorch')
                compiled_graph_output = compiled_graph(data.M, data.x1[i], data.y1[j])
                self.assertEqual(inductor_graph_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_zero_input(self, dtype):
        data = Test_Data(dtype)
        model = CustomModelMMAdd1().eval()
        model_output = model(
            data.M * 0, data.x1[0] * 0, data.y1[0] * 0)
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(
            data.M * 0, data.x1[0] * 0, data.y1[0] * 0)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_inf_input(self, dtype):
        data = Test_Data(dtype)
        model = CustomModelMMAdd1().eval()
        model_output = model(
            data.M / 0, data.x1[0] / 0, data.y1[0] / 0)
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(
            data.M / 0, data.x1[0] / 0, data.y1[0] / 0)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_nan_input(self, dtype):
        data = Test_Data(dtype)
        model = CustomModelMMAdd1().eval()
        model_output = model(
            data.M * float('nan'), data.x1[0] * float('nan'),
            data.y1[0] * float('nan'))
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(
            data.M * float('nan'), data.x1[0] * float('nan'),
            data.y1[0] * float('nan'))
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_identity_input_nan(self, dtype):
        data = Test_Data(dtype)
        if dtype == 'bfloat16':
            self.skipTest("Skipping it since this testcase is not applicable for BF16.")
        model = CustomModelMMAdd1().eval()
        model_output = model(
            torch.eye(data.M.shape[0], data.M.shape[1]),
            data.x1[0] * float('nan'), data.y1[0] * float('nan'))
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(
            torch.eye(data.M.shape[0], data.M.shape[1]),
            data.x1[0] * float('nan'), data.y1[0] * float('nan'))
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TestADDMM_GELU(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_zendnn_addmm_gelu(self, dtype):
        if dtype == 'bfloat16':
            self.skipTest("Skipping it due to issue BF16 path.")
        data = Test_Data(dtype)
        model = CustomModelAddmmGelu1().eval()
        for i in range(len(data.x1)):
            for j in range(len(data.y1)):
                model_output = model(data.M, data.x1[i], data.y1[j])
                compiled_graph = torch.compile(model, backend='zentorch')
                compiled_graph_output = compiled_graph(data.M, data.x1[i], data.y1[j])
                self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_zendnn_addmm_gelu_tanh(self, dtype):
        if dtype == 'bfloat16':
            self.skipTest("Skipping it due to issue BF16 path.")
        data = Test_Data(dtype)
        model = CustomModelAddmmGelu2().eval()
        for i in range(len(data.x1)):
            for j in range(len(data.y1)):
                model_output = model(data.M, data.x1[i], data.y1[j])
                compiled_graph = torch.compile(model, backend='zentorch')
                compiled_graph_output = compiled_graph(data.M, data.x1[i], data.y1[j])
                self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TestADDMM_RELU(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_zendnn_addmm_relu(self, dtype):
        if dtype == 'bfloat16':
            self.skipTest("Skipping it due to issue BF16 path.")
        data = Test_Data(dtype)
        model = CustomModelAddmmRelu2().eval()
        for i in range(len(data.x1)):
            for j in range(len(data.y1)):
                model_output = model(data.M, data.x1[i], data.y1[j])
                compiled_graph = torch.compile(model, backend='zentorch')
                compiled_graph_output = compiled_graph(data.M, data.x1[i], data.y1[j])
                self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_custom_addmm_relu1(self, dtype):
        data = Test_Data(dtype)
        model = CustomModelAddmmReLU1(data.n, data.n - 2).eval()
        if dtype == 'bfloat16':
            model = model.bfloat16()
        model_output = model(data.input)
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(data.input)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_custom_addmm_relu1_with_nan_or_inf(self, dtype):
        if dtype == 'bfloat16':
            self.skipTest("Skipping it since this testcase is not applicable for BF16.")
        data = Test_Data(dtype)
        model = CustomModelAddmmReLU1(data.n, data.n - 2).eval()
        data.input[0][0] = float('nan')
        data.input[1][1] = float('inf')
        inductor_graph = torch.compile(model, backend='inductor')
        inductor_graph_output = inductor_graph(data.input)
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(data.input)
        self.assertEqual(inductor_graph_output, compiled_graph_output)


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TestLinear_Relu(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_zendnn_linear_relu(self, dtype):
        data = Test_Data(dtype)
        model = nn.Sequential(nn.Linear(data.n, data.m), nn.ReLU())
        if dtype == 'bfloat16':
            model = model.bfloat16()
        fx_g = make_fx(model)(data.input)
        fx_g_modified = zentorch.optimize(fx_g)
        fx_g_output = fx_g(data.input)
        fx_g_modified_output = fx_g_modified(data.input)
        self.assertEqual(fx_g_output, fx_g_modified_output)
        for node in fx_g_modified.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                if node.target.name() in ["aten::addmm"]:
                    self.assertEqual(node.target, torch.ops.zentorch.zendnn_addmm)


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TestLinear_Gelu(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_zendnn_linear_gelu_tanh(self, dtype):
        data = Test_Data(dtype)
        model = nn.Sequential(nn.Linear(data.n, data.m), nn.GELU(approximate="tanh"))
        if dtype == 'bfloat16':
            model = model.bfloat16()
        model_output = model(data.input)
        compiled_graph = torch.compile(model, backend='inductor')
        compiled_graph_output = compiled_graph(data.input)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_zendnn_linear_gelu_none(self, dtype):
        data = Test_Data(dtype)
        model = nn.Sequential(nn.Linear(data.n, data.m), nn.GELU(approximate="none"))
        if dtype == 'bfloat16':
            model = model.bfloat16()
        model_output = model(data.input)
        compiled_graph = torch.compile(model, backend='zentorch')
        compiled_graph_output = compiled_graph(data.input)
        self.assertEqual(model_output, compiled_graph_output)


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TestBMMADD(TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.no_grad()
    def test_zendnn_bmm_baddbmm(self, dtype):
        if dtype == 'bfloat16':
            self.skipTest("Skipping it due to issue with BF16 path.")
        data = Test_Data(dtype)
        model = CustomModelBMMAdd1().eval()
        for i in range(len(data.x2)):
            for j in range(len(data.y2)):
                model_output = model(data.M2, data.x2[i], data.y2[j])
                compiled_graph = torch.compile(model, backend='zentorch')
                compiled_graph_output = compiled_graph(data.M2, data.x2[i], data.y2[j])
                self.assertEqual(
                    model_output, compiled_graph_output, atol=1e-1, rtol=1e-3
                )


if __name__ == "__main__":
    run_tests()
