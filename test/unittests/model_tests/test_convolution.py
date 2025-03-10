# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    ConvTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    freeze_opt,
    test_with_freeze_opt,
    counters,
)
from zentorch._compile_backend import conv_config  # noqa: E402
# set zentorch_conv replacement to true
conv_config.enable_zentorch_conv(True)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Convolution(nn.Module):
    def __init__(self):
        super(Custom_Model_Convolution, self).__init__()
        self.convolution = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        # Convert weights to channels last memory format, zentorch requires this in op-replacement check
        self.convolution.weight.data = self.convolution.weight.data.to(memory_format=torch.channels_last)

    def forward(self, input):
        output = self.convolution(input)
        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Convolution_Model(ConvTestCase):
    @ConvTestCase.hypothesis_params_conv_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_convolution_model(self, dtype, freeze_opt):
        model = Custom_Model_Convolution()
        if dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        model_output = model(self.data.conv_input)
        reset_dynamo()
        zentorch_graph = torch.compile(model, backend="zentorch", dynamic=False)
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_convolution"], 0)
        zentorch_graph_output = test_with_freeze_opt(
            zentorch_graph,
            (self.data.conv_input),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["zentorch_convolution"], 1)
        self.assertEqual(
            model_output,
            zentorch_graph_output,
        )


if __name__ == "__main__":
    run_tests()
