#******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#******************************************************************************

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
class TestZENDNN(TestCase):

    @torch.no_grad()
    def test_embedding_bag_zendnn(self):
        R = torch.randint(11, 20, (1,)).item()
        W = torch.randint(1, 15, (1,)).item()
        embedding_matrix = torch.rand(R, 3)
        input = torch.randint(0, R, (W,))
        offsets = torch.tensor([0, W])

        y_eb, _, _, _ = torch._C._VariableFunctions._embedding_bag(embedding_matrix, input, offsets, False, 0, False, None, False)

        y_ebz, _, _, _ = zentorch._C.embedding_bag_zendnn(embedding_matrix, input, offsets, False, 0, False, None, False, -1)

        self.assertEqual(y_eb, y_ebz)

    @unittest.skipIf(not zentorch._C.is_bf16_supported(), "CPU has does not support AVX512 BF16.")
    def test_bf16_device(self):
        self.assertTrue(zentorch._C.is_bf16_supported(), "CPU supports AVX512 BF16.")

class SampleEmbeddingNN(nn.Module):
        def __init__(self, embedding_dim, output_dim):
            super(SampleEmbeddingNN, self).__init__()
            self.embedding = nn.EmbeddingBag(10000, embedding_dim)
            self.intermediate = nn.Linear(embedding_dim,output_dim)
            self.output = nn.Linear(output_dim, 1)

        def forward(self, input):
            embed = self.embedding(input)
            intermediate = self.intermediate(embed)
            output = self.output(intermediate)
            return output

class TestZenDNNOptimize(TestCase):

    @torch.no_grad()
    def test_zendnn_optimize_function(self):
        model = SampleEmbeddingNN(100,10)
        input = torch.randint(0,10000, (1,10))

        fx_g = make_fx(model)(input)
        fx_g_modified = zentorch.optimize(fx_g)

        fx_g_output = fx_g(input)
        fx_g_modified_output = fx_g_modified(input)

        self.assertAlmostEqual(fx_g_output.item(), fx_g_modified_output.item())

if __name__ == '__main__':
    run_tests()
