# ******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import unittest
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx

try:
    import torch_zendnn_plugin as zentorch

    HAS_PT_PLUGIN = True
except ImportError:
    HAS_PT_PLUGIN = False


@unittest.skipIf(not HAS_PT_PLUGIN, "PT PLUGIN is not installed")
class TestZENDNNOps(TestCase):
    @torch.no_grad()
    def test_embedding_bag_zendnn(self):
        R = torch.randint(11, 20, (1,)).item()
        W = torch.randint(1, 15, (1,)).item()
        embedding_matrix = torch.rand(R, 3)
        input = torch.randint(0, R, (W,))
        offsets = torch.tensor([0, W])

        y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(
            embedding_matrix, input, offsets, False, 0, False, None, False
        )

        y_ebz, _, _, _ = torch.ops.zentorch.zendnn_embedding_bag(
            embedding_matrix, input, offsets, False, 0, False, None, False, -1
        )

        self.assertEqual(y_eb, y_ebz)

    @unittest.skipIf(
        not zentorch._C.is_bf16_supported(), "CPU does not support AVX512 BF16."
    )
    def test_bf16_device(self):
        self.assertTrue(zentorch._C.is_bf16_supported(), "CPU supports AVX512 BF16.")

    @torch.no_grad()
    def test_zendnn_matmul(self):
        b = torch.randint(1, 11, (1,)).item()
        m = torch.randint(1, 11, (1,)).item()
        k = torch.randint(1, 11, (1,)).item()
        n = torch.randint(1, 11, (1,)).item()

        # m*k, k*n, m*n
        x = torch.randn(m, k)
        y = torch.randn(k, n)
        input = torch.randn(m, n)

        # b*m*k, b*k*n, b*m*n
        x3d = torch.randn(b, m, k)
        y3d = torch.randn(b, k, n)
        input3d = torch.randn(b, m, n)

        # mm
        self.assertEqual(
            torch._C._VariableFunctions.mm(x, y), torch.ops.zentorch.zendnn_mm(x, y)
        )
        # mm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(torch._C._VariableFunctions.mm(x, y)),
            torch.ops.zentorch.zendnn_mm(x, y, fuse=1),
        )

        # addmm
        self.assertEqual(
            torch._C._VariableFunctions.addmm(input, x, y),
            torch.ops.zentorch.zendnn_addmm(
                input,
                x,
                y,
            ),
        )
        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(input, x, y, beta=1.3),
            torch.ops.zentorch.zendnn_addmm(input, x, y, beta=1.3),
        )
        # addmm->relu [used kw_only arguments for fuse relu]
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(input, x, y, beta=1.5, alpha=1.7)
            ),
            torch.ops.zentorch.zendnn_addmm(input, x, y, beta=1.5, alpha=1.7, fuse=1),
        )

        # bmm
        self.assertEqual(
            torch._C._VariableFunctions.bmm(x3d, y3d),
            torch.ops.zentorch.zendnn_bmm(x3d, y3d),
        )

        # baddbmm
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(input3d, x3d, y3d),
            torch.ops.zentorch.zendnn_baddbmm(input3d, x3d, y3d),
        )

    @unittest.skipIf(
        not zentorch._C.is_bf16_supported(), "CPU does not support AVX512 BF16."
    )
    @torch.no_grad()
    def test_zendnn_matmul_bf16(self):
        b = torch.randint(1, 11, (1,)).item()
        m = torch.randint(1, 11, (1,)).item()
        k = torch.randint(1, 11, (1,)).item()
        n = torch.randint(1, 11, (1,)).item()

        # m*k, k*n, m*n
        x = torch.randn(m, k).type(torch.bfloat16)
        y = torch.randn(k, n).type(torch.bfloat16)
        input = torch.randn(m, n).type(torch.bfloat16)

        # b*m*k, b*k*n, b*m*n
        x3d = torch.randn(b, m, k).type(torch.bfloat16)
        y3d = torch.randn(b, k, n).type(torch.bfloat16)
        input3d = torch.randn(b, m, n).type(torch.bfloat16)

        # mm
        self.assertEqual(
            torch._C._VariableFunctions.mm(x, y),
            torch.ops.zentorch.zendnn_mm(x, y),
            atol=1e-1,
            rtol=1e-3,
        )
        # mm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(torch._C._VariableFunctions.mm(x, y)),
            torch.ops.zentorch.zendnn_mm(x, y, fuse=1),
            atol=1e-1,
            rtol=1e-3,
        )

        # addmm
        self.assertEqual(
            torch._C._VariableFunctions.addmm(input, x, y),
            torch.ops.zentorch.zendnn_addmm(
                input,
                x,
                y,
            ),
            atol=1e-1,
            rtol=1e-3,
        )
        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(input, x, y, beta=1),
            torch.ops.zentorch.zendnn_addmm(input, x, y, beta=1),
            atol=1e-1,
            rtol=1e-3,
        )
        # addmm->relu [used kw_only arguments for fuse relu]
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(input, x, y, beta=1.5, alpha=1.7)
            ),
            torch.ops.zentorch.zendnn_addmm(input, x, y, beta=1.5, alpha=1.7, fuse=1),
            atol=1e-1,
            rtol=1e-3,
        )

        # bmm
        self.assertEqual(
            torch._C._VariableFunctions.bmm(x3d, y3d),
            torch.ops.zentorch.zendnn_bmm(x3d, y3d),
            atol=1e-1,
            rtol=1e-3,
        )

        # baddbmm
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(input3d, x3d, y3d),
            torch.ops.zentorch.zendnn_baddbmm(input3d, x3d, y3d),
            atol=1e-1,
            rtol=1e-3,
        )


class SampleEmbeddingNN(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(SampleEmbeddingNN, self).__init__()
        self.embedding = nn.EmbeddingBag(10000, embedding_dim)
        self.intermediate = nn.Linear(embedding_dim, output_dim)
        self.output = nn.Linear(output_dim, 1)

    def forward(self, input):
        embed = self.embedding(input)
        intermediate = self.intermediate(embed)
        output = self.output(intermediate)
        return output


class SampleGroupEmbeddingBag(nn.Module):
    def __init__(self):
        super(SampleGroupEmbeddingBag, self).__init__()
        self.eb_bags_grp_0 = [
            torch.nn.EmbeddingBag(5, 14, mode="sum") for _ in range(5)
        ]
        self.eb_bags_grp_1 = [
            torch.nn.EmbeddingBag(5, 14, mode="sum") for _ in range(10)
        ]
        self.eb_bags_grp_2 = [
            torch.nn.EmbeddingBag(5, 14, mode="sum") for _ in range(5)
        ]
        self.mm_0 = torch.matmul
        self.mm_1 = torch.matmul
        self.mm_2 = torch.matmul
        self.mm_3 = torch.matmul

    def forward(self, mm_0_a, mm_0_b, eb_input, eb_offset):
        mm_0_output = self.mm_0(mm_0_a, mm_0_b)

        eb_outputs_grp_0 = [
            self.eb_bags_grp_0[i](eb_input, eb_offset) for i in range(5)
        ]
        eb_sum_0 = torch.unsqueeze(torch.sum(torch.cat(eb_outputs_grp_0), dim=0), dim=0)

        mm_1_output = self.mm_1(mm_0_output, eb_sum_0)

        eb_outputs_grp_1 = [
            self.eb_bags_grp_1[i](eb_input, eb_offset) for i in range(10)
        ]
        eb_sum_1 = torch.unsqueeze(torch.sum(torch.cat(eb_outputs_grp_1), dim=0), dim=0)

        mm_2_output = self.mm_2(eb_sum_1, mm_1_output)

        eb_outputs_grp_2 = [
            self.eb_bags_grp_2[i](eb_input, eb_offset) for i in range(5)
        ]
        eb_sum_2 = torch.unsqueeze(torch.sum(torch.cat(eb_outputs_grp_2), dim=0), dim=0)

        mm_3_output = self.mm_3(eb_sum_2, torch.transpose(mm_2_output, 0, 1))

        return mm_3_output


class BmmAdd(nn.Module):
    def __init__(self):
        super(BmmAdd, self).__init__()

    def forward(self, input, batch1, batch2):
        bmm_res = torch.bmm(batch1, batch2)
        add_res = torch.add(bmm_res, input)
        baddbmm_res = torch.baddbmm(add_res, batch1, batch2, beta=1.5, alpha=1.4)
        return baddbmm_res


class AddmmRelu(nn.Module):
    def __init__(self):
        super(AddmmRelu, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        # Relu
        relu1_res = torch.relu(add_res)
        addmm_res = torch.addmm(relu1_res, batch1, batch2, beta=1.7, alpha=1.6)
        # inplace Relu
        relu2_res = torch.relu_(addmm_res)
        return relu2_res


class AddmmGeluTanh(nn.Module):
    def __init__(self):
        super(AddmmGeluTanh, self).__init__()
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        GELU1_res = self.gelu(add_res)
        addmm_res = torch.addmm(GELU1_res, batch1, batch2)
        GELU2_res = self.gelu(addmm_res)
        return GELU2_res


class AddmmGelu(nn.Module):
    def __init__(self):
        super(AddmmGelu, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        GELU1_res = nn.functional.gelu(add_res, approximate="tanh")
        addmm_res = torch.addmm(GELU1_res, batch1, batch2, beta=1.7, alpha=1.6)
        GELU2_res = torch._C._nn.gelu_(addmm_res, approximate="none")
        return GELU2_res


class TestZenDNNOptimize(TestCase):
    @torch.no_grad()
    def test_zentorch_compile_function(self):
        model = SampleEmbeddingNN(100, 10)
        input = torch.randint(0, 10000, (1, 10))

        compiled_graph = torch.compile(model, backend="zentorch")

        model_output = model(input)
        compiled_graph_out = compiled_graph(input)

        self.assertAlmostEqual(model_output.item(), compiled_graph_out.item())

    @torch.no_grad()
    def test_zendnn_optimize_function(self):
        model = SampleEmbeddingNN(100, 10)
        input = torch.randint(0, 10000, (1, 10))

        fx_g = make_fx(model)(input)
        fx_g_modified = zentorch.optimize(fx_g)

        fx_g_output = fx_g(input)
        fx_g_modified_output = fx_g_modified(input)

        self.assertAlmostEqual(fx_g_output.item(), fx_g_modified_output.item())

    @torch.no_grad()
    def test_zentorch_compile_linear_relu(self):
        model = nn.Sequential(nn.Linear(4, 5), nn.ReLU())

        input = torch.randn(10, 4)

        model_output = model(input)

        compiled_graph = torch.compile(model, backend="zentorch")

        compiled_graph_out = compiled_graph(input)

        self.assertEqual(model_output, compiled_graph_out)

    @torch.no_grad()
    def test_zendnn_linear_relu(self):
        model = nn.Sequential(nn.Linear(4, 5), nn.ReLU())

        input = torch.randn(10, 4)

        fx_g = make_fx(model)(input)

        fx_g_output = fx_g(input)

        fx_g_modified = zentorch.optimize(fx_g)

        fx_g_modified_output = fx_g_modified(input)

        self.assertEqual(fx_g_output, fx_g_modified_output)

        for node in fx_g_modified.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                if node.target.name() in ["aten::addmm"]:
                    self.assertEqual(node.target, torch.ops.zentorch.zendnn_addmm)

    @torch.no_grad()
    def test_zentorch_compile_bmm_baddbmm(self):
        M = torch.randn(60, 30, 50)

        x1 = [
            torch.randn(60, 30, 40),
            torch.randn(60, 40, 30).transpose(1, 2),
            torch.randn(30, 60, 40).transpose(0, 1),
        ]

        y1 = [
            torch.randn(60, 40, 50),
            torch.randn(60, 50, 40).transpose(1, 2),
            torch.randn(50, 40, 60).transpose(0, 2),
        ]

        model = BmmAdd().eval()
        compiled_graph = torch.compile(model, backend="zentorch")
        for i in range(len(x1)):
            for j in range(len(y1)):
                model_output = model(M, x1[i], y1[j])
                compiled_graph_out = compiled_graph(M, x1[i], y1[j])

                self.assertEqual(model_output, compiled_graph_out, atol=1e-1, rtol=1e-3)

    @torch.no_grad()
    def test_zendnn_bmm_baddbmm(self):
        M = torch.randn(60, 30, 50)

        x1 = [
            torch.randn(60, 30, 40),
            torch.randn(60, 40, 30).transpose(1, 2),
            torch.randn(30, 60, 40).transpose(0, 1),
        ]

        y1 = [
            torch.randn(60, 40, 50),
            torch.randn(60, 50, 40).transpose(1, 2),
            torch.randn(50, 40, 60).transpose(0, 2),
        ]

        model = BmmAdd().eval()
        for i in range(len(x1)):
            for j in range(len(y1)):
                fx_g = make_fx(model)(M, x1[i], y1[j])

                fx_g_output = fx_g(M, x1[i], y1[j])

                fx_g_modified = zentorch.optimize(fx_g)

                fx_g_modified_output = fx_g_modified(M, x1[i], y1[j])

                self.assertEqual(
                    fx_g_output, fx_g_modified_output, atol=1e-1, rtol=1e-3
                )

                for node in fx_g_modified.graph.nodes:
                    if isinstance(node.target, torch._ops.OpOverload):
                        if node.target.name() in ["aten::bmm", "aten::baddbmm"]:
                            self.assertEqual(
                                node.target,
                                torch.ops.zentorch.zendnn_bmm
                                or torch.ops.zentorch.zendnn_baddbmm,
                            )

    @torch.no_grad()
    def test_zentorch_compile_addmm_relu(self):
        M = torch.randn(60, 30)

        x1 = [
            torch.randn(60, 40),
            torch.randn(40, 60).transpose(0, 1),
        ]

        y1 = [
            torch.randn(40, 30),
            torch.randn(30, 40).transpose(1, 0),
        ]
        model = AddmmRelu().eval()
        compiled_graph = torch.compile(model, backend="zentorch")
        for i in range(len(x1)):
            for j in range(len(y1)):
                model_output = model(M, x1[i], y1[j])

                compiled_graph_out = compiled_graph(M, x1[i], y1[j])

                self.assertEqual(model_output, compiled_graph_out, atol=1e-1, rtol=1e-3)

    @torch.no_grad()
    def test_zendnn_addmm_relu(self):
        M = torch.randn(60, 30)

        x1 = [
            torch.randn(60, 40),
            torch.randn(40, 60).transpose(0, 1),
        ]

        y1 = [
            torch.randn(40, 30),
            torch.randn(30, 40).transpose(1, 0),
        ]
        model = AddmmRelu().eval()
        for i in range(len(x1)):
            for j in range(len(y1)):
                fx_g = make_fx(model)(M, x1[i], y1[j])

                fx_g_output = fx_g(M, x1[i], y1[j])

                fx_g_modified = zentorch.optimize(fx_g)

                fx_g_modified_output = fx_g_modified(M, x1[i], y1[j])

                self.assertEqual(
                    fx_g_output, fx_g_modified_output, atol=1e-1, rtol=1e-3
                )
                for node in fx_g_modified.graph.nodes:
                    if isinstance(node.target, torch._ops.OpOverload):
                        if node.target.name() in ["aten::mm", "aten::addmm"]:
                            self.assertEqual(
                                node.target,
                                torch.ops.zentorch.zendnn_mm
                                or torch.ops.zentorch.zendnn_addmm,
                            )

    @torch.no_grad()
    def test_zentorch_compile_addmm_gelu(self):
        M = torch.randn(60, 30)

        x1 = [
            torch.randn(60, 40),
            torch.randn(40, 60).transpose(0, 1),
        ]

        y1 = [
            torch.randn(40, 30),
            torch.randn(30, 40).transpose(1, 0),
        ]

        model1 = AddmmGelu().eval()
        model2 = AddmmGeluTanh().eval()
        model = [model1, model2]

        compiled_model1 = torch.compile(model1, backend="zentorch")
        compiled_model2 = torch.compile(model2, backend="zentorch")
        compiled_models = [compiled_model1, compiled_model2]

        for m, c_m in zip(model, compiled_models):
            for i in range(len(x1)):
                for j in range(len(y1)):
                    model_output = m(M, x1[i], y1[j])

                    compiled_graph_out = c_m(M, x1[i], y1[j])

                    self.assertEqual(model_output, compiled_graph_out)

    @torch.no_grad()
    def test_zendnn_addmm_gelu(self):
        M = torch.randn(60, 30)

        x1 = [
            torch.randn(60, 40),
            torch.randn(40, 60).transpose(0, 1),
        ]

        y1 = [
            torch.randn(40, 30),
            torch.randn(30, 40).transpose(1, 0),
        ]

        model1 = AddmmGelu().eval()
        model2 = AddmmGeluTanh().eval()
        model = [model1, model2]
        for m in model:
            for i in range(len(x1)):
                for j in range(len(y1)):
                    fx_g = make_fx(m)(M, x1[i], y1[j])

                    fx_g_output = fx_g(M, x1[i], y1[j])

                    fx_g_modified = zentorch.optimize(fx_g)

                    fx_g_modified_output = fx_g_modified(M, x1[i], y1[j])
                    self.assertEqual(fx_g_output, fx_g_modified_output)
                    for node in fx_g_modified.graph.nodes:
                        if isinstance(node.target, torch._ops.OpOverload):
                            if node.target.name() in ["aten::mm", "aten::addmm"]:
                                self.assertEqual(
                                    node.target,
                                    torch.ops.zentorch.zendnn_mm
                                    or torch.ops.zentorch.zendnn_addmm,
                                )

    @torch.no_grad()
    def test_zentorch_compile_linear_gelu(self):
        model1 = nn.Sequential(nn.Linear(40, 50), nn.GELU(approximate="none"))

        model2 = nn.Sequential(nn.Linear(40, 50), nn.GELU(approximate="tanh"))

        input = torch.randn(10, 40)

        model = [model1, model2]

        compiled_model1 = torch.compile(model1, backend="zentorch")
        compiled_model2 = torch.compile(model2, backend="zentorch")
        compiled_models = [compiled_model1, compiled_model2]

        for m, c_m in zip(model, compiled_models):
            model_output = m(input)

            compiled_graph_out = c_m(input)

            self.assertEqual(model_output, compiled_graph_out)

    @torch.no_grad()
    def test_zendnn_linear_gelu(self):
        model1 = nn.Sequential(nn.Linear(40, 50), nn.GELU(approximate="none"))

        model2 = nn.Sequential(nn.Linear(40, 50), nn.GELU(approximate="tanh"))

        input = torch.randn(10, 40)

        model = [model1, model2]

        for m in model:
            fx_g = make_fx(m)(input)

            fx_g_output = fx_g(input)

            fx_g_modified = zentorch.optimize(fx_g)

            fx_g_modified_output = fx_g_modified(input)

            self.assertEqual(fx_g_output, fx_g_modified_output)

            for node in fx_g_modified.graph.nodes:
                if isinstance(node.target, torch._ops.OpOverload):
                    if node.target.name() in ["aten::addmm"]:
                        self.assertEqual(node.target, torch.ops.zentorch.zendnn_addmm)

    @torch.no_grad()
    def test_group_embeddingbag(self):
        model = SampleGroupEmbeddingBag()
        x = {
            "mm_0": {"a": torch.randn(14, 14), "b": torch.randn(14, 1)},
            "eb_bags": {"input": torch.randint(0, 4, (5, 14)), "offset": None},
            "last_input": torch.randn(14, 1),
        }

        fx_g = make_fx(model)(
            x["mm_0"]["a"],
            x["mm_0"]["b"],
            x["eb_bags"]["input"],
            x["eb_bags"]["offset"],
        )
        fx_g_output = fx_g(
            x["mm_0"]["a"],
            x["mm_0"]["b"],
            x["eb_bags"]["input"],
            x["eb_bags"]["offset"],
        )

        fx_g_optimized = zentorch.optimize(fx_g)
        fx_g_optimized = zentorch._optimize.replace_emb_bag(fx_g_optimized)
        fx_g_optimized_output = fx_g_optimized(
            x["mm_0"]["a"],
            x["mm_0"]["b"],
            x["eb_bags"]["input"],
            x["eb_bags"]["offset"],
        )

        self.assertAlmostEqual(fx_g_output.item(), fx_g_optimized_output.item())


if __name__ == "__main__":
    run_tests()
