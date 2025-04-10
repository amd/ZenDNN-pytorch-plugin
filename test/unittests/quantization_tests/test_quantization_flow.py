# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
import zentorch  # noqa: F401
from pathlib import Path
from torch.ao.quantization.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
)
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from typing import Dict


sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
)


class Custom_Model_Simple_Test_MLP(torch.nn.Module):
    """A minimal DLRM-like model for quantization replacement tests."""

    def __init__(self, dense_dim=10, emb_dim=4, vocab_size=10, top_dim=4, bias=True):
        super().__init__()
        # Bottom MLP: dense features
        self.bot_mlp = torch.nn.Sequential(
            torch.nn.Linear(dense_dim, emb_dim, bias=bias),
            torch.nn.ReLU(),
        )
        # Single embedding table for sparse features
        self.emb = torch.nn.EmbeddingBag(vocab_size, emb_dim, mode="sum")
        # Top MLP: after feature interaction
        self.top_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, top_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(top_dim, 1, bias=bias),
            torch.nn.Sigmoid(),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, dense_x, offsets, indices):
        # dense_x: [batch, dense_dim]
        # indices: [num_indices]
        # offsets: [batch + 1]
        dense_out = self.bot_mlp(dense_x)  # [batch, emb_dim]
        emb_out = self.emb(indices, offsets)  # [batch, emb_dim]
        # Simple feature interaction: concatenate dense and sparse
        z = torch.cat([dense_out, emb_out], dim=1)
        sigmoid = self.sigmoid(z)
        return sigmoid


class Custom_Model_Simple_Test_MLP_3D(torch.nn.Module):
    """A simple MLP model that processes 3D inputs."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5, bias=True):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim, bias=bias),
            torch.nn.Sigmoid(),
        )

    def forward(self, x_3d):
        # x_3d: [batch, seq_len, input_dim]
        return self.mlp(x_3d)


class Simple_Quantized_Model_Test(Zentorch_TestCase):
    """Simple test case for quantized models"""
    def setUp(self, bias=True):
        super().setUp()

        self.model = Custom_Model_Simple_Test_MLP(10, 20, bias=bias)
        batch = 2
        dense_dim = 10
        self.input = (
            torch.randn(batch, dense_dim, dtype=torch.float32),  # dense_x
            torch.tensor([0, 2], dtype=torch.long),              # offsets
            torch.tensor([1, 2, 3, 4], dtype=torch.long),        # indices
        )

    def execute_test(self):
        # Quantize model
        quantized_model = self.quantize()

        # Model output before zentorch compilation
        output_before = quantized_model(self.input[0], self.input[1], self.input[2])

        # Compile with Zentorch to replace ops
        quantized_model = torch.compile(quantized_model, backend="zentorch")

        # Model output after compilation/op replacement
        output_after = quantized_model(self.input[0], self.input[1], self.input[2])

        # Compare outputs before and after zentorch compilation
        self.assertEqual(output_before, output_after)

        # Assert that the compilation pass has replaced Q/DQ ops
        self.assertTrue(
            expr=self.were_ops_replaced(quantized_model),
            msg="were_ops_replaced() not implemented or ops were not replaced "
                "by Zentorch compilation.",
        )

    def were_ops_replaced(self, quantized_model):
        return False

    def quantize(self):
        pass


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Pt2e_Quantization_Replacement(Simple_Quantized_Model_Test):
    """Tests X86 Inductor quantization replacement with 2D inputs."""
    def test_pt2e_quantization_replacement_flow(self):
        self.execute_test()

    def quantize(self):
        # Export model for quantization
        quantized_model = torch.export.export_for_training(
            self.model,
            self.input,
        ).module()

        # Quantize model using PT2E
        x86_quantizer = X86InductorQuantizer()
        x86_quantizer.set_global(self.get_amd_x86_quantization_configuration())

        quantized_model = prepare_pt2e(quantized_model, x86_quantizer)
        quantized_model = convert_pt2e(
            quantized_model,
            use_reference_representation=False,
            fold_quantize=True,
        )

        return quantized_model

    def were_ops_replaced(self, quantized_model):
        qdq_ops = {"quantize_per_tensor", "dequantize_per_tensor"}

        # Check for the presence of Q/DQ ops
        return all(
            not (
                node.op == 'call_function' and node.target in qdq_ops
            )
            for node in quantized_model.graph.nodes
        )

    def get_amd_x86_quantization_configuration(self):
        act_extra_args: Dict[str, any] = {
            "eps": 2**-12,
            "qscheme": torch.per_tensor_affine,
            "bins": 256,
            "reduce_range": False,
        }
        act_observer = HistogramObserver
        act_quantization_spec = QuantizationSpec(
            dtype=torch.uint8,
            quant_min=0,
            quant_max=255,
            qscheme=torch.per_tensor_affine,
            is_dynamic=False,
            observer_or_fake_quant_ctr=act_observer.with_args(
                **act_extra_args
            ),
        )
        weight_extra_args: Dict[str, any] = {
            "eps": 2**-12,
            "qscheme": torch.per_channel_symmetric,
        }
        weight_observer: _ObserverOrFakeQuantizeConstructor = (
            PerChannelMinMaxObserver
        )
        weight_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            ch_axis=0,
            qscheme=torch.per_channel_symmetric,
            is_dynamic=False,
            observer_or_fake_quant_ctr=weight_observer.with_args(
                **weight_extra_args
            ),
        )
        bias_quantization_spec = None
        quantization_config = QuantizationConfig(
            act_quantization_spec,
            act_quantization_spec,
            weight_quantization_spec,
            bias_quantization_spec,
            False,
        )

        return quantization_config


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Pt2e_Quantization_Replacement_3D(Simple_Quantized_Model_Test):
    """Tests X86 Inductor quantization replacement with 3D inputs."""
    def setUp(self):
        super().setUp()

        self.input_dim = 10
        self.hidden_dim = 15
        self.output_dim = 8
        self.batch_size = 2
        self.seq_len = 3

        self.model = Custom_Model_Simple_Test_MLP_3D(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        # Single 3D input tensor
        self.input_tensor = torch.randn(
            self.batch_size, self.seq_len, self.input_dim, dtype=torch.float32
        )
        # execute_test expects self.input to be a tuple
        self.input = (self.input_tensor,)

    def execute_test(self):
        # Quantize model
        quantized_model = self.quantize()

        # Model output before zentorch compilation
        # Input is a tuple, so unpack it
        output_before = quantized_model(*self.input)

        # Compile with Zentorch to replace ops
        quantized_model_compiled = torch.compile(quantized_model, backend="zentorch")

        # Model output after compilation/op replacement
        output_after = quantized_model_compiled(*self.input)

        # Compare outputs before and after zentorch compilation
        self.assertEqual(output_before, output_after)

        # Assert that the compilation pass has replaced Q/DQ ops
        self.assertTrue(
            expr=self.were_ops_replaced(quantized_model_compiled),
            msg="were_ops_replaced() not implemented or ops were not replaced "
                "by Zentorch compilation for 3D model.",
        )

    def quantize(self):
        quantized_model = torch.export.export_for_training(
            self.model,
            self.input,
        ).module()

        # Quantize model using PT2E X86InductorQuantizer
        x86_quantizer = X86InductorQuantizer()
        quant_config = Test_Pt2e_Quantization_Replacement().get_amd_x86_quantization_configuration()
        x86_quantizer.set_global(quant_config)

        quantized_model = prepare_pt2e(quantized_model, x86_quantizer)
        # Calibrate - run with example data
        quantized_model(*self.input)

        quantized_model = convert_pt2e(
            quantized_model,
            use_reference_representation=False,
            fold_quantize=True,
        )
        return quantized_model

    def were_ops_replaced(self, quantized_model):
        qdq_ops = {"quantize_per_tensor", "dequantize_per_tensor"}

        # Check for the presence of Q/DQ ops
        return all(
            not (
                node.op == 'call_function' and node.target in qdq_ops
            )
            for node in quantized_model.graph.nodes
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Pt2e_Quantization_No_Bias_Replacement(Simple_Quantized_Model_Test):
    """Tests X86 Inductor quantization replacement with 2D inputs and no Bias."""
    def test_pt2e_quantization_replacement_flow(self):
        self.execute_test()

    def setUp(self):
        super().setUp(bias=False)

    def quantize(self):
        # Export model for quantization
        quantized_model = torch.export.export_for_training(
            self.model,
            self.input,
        ).module()

        # Quantize model using PT2E
        x86_quantizer = X86InductorQuantizer()
        quant_config = Test_Pt2e_Quantization_Replacement().get_amd_x86_quantization_configuration()
        x86_quantizer.set_global(quant_config)

        quantized_model = prepare_pt2e(quantized_model, x86_quantizer)
        quantized_model = convert_pt2e(
            quantized_model,
            use_reference_representation=False,
            fold_quantize=True,
        )

        return quantized_model

    def were_ops_replaced(self, quantized_model):
        qdq_ops = {"quantize_per_tensor", "dequantize_per_tensor"}

        # Check for the presence of Q/DQ ops
        return all(
            not (
                node.op == 'call_function' and node.target in qdq_ops
            )
            for node in quantized_model.graph.nodes
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Pt2e_Quantization_No_Bias_Replacement_3D(Simple_Quantized_Model_Test):
    """Tests X86 Inductor quantization replacement with 3D inputs."""
    def setUp(self):
        super().setUp(bias=False)

        self.input_dim = 10
        self.hidden_dim = 15
        self.output_dim = 8
        self.batch_size = 2
        self.seq_len = 3

        self.model = Custom_Model_Simple_Test_MLP_3D(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            bias=False,
        )
        # Single 3D input tensor
        self.input_tensor = torch.randn(
            self.batch_size, self.seq_len, self.input_dim, dtype=torch.float32
        )
        # execute_test expects self.input to be a tuple
        self.input = (self.input_tensor,)

    def execute_test(self):
        # Quantize model
        quantized_model = self.quantize()

        # Model output before zentorch compilation
        # Input is a tuple, so unpack it
        output_before = quantized_model(*self.input)

        # Compile with Zentorch to replace ops
        quantized_model_compiled = torch.compile(quantized_model, backend="zentorch")

        # Model output after compilation/op replacement
        output_after = quantized_model_compiled(*self.input)

        # Compare outputs before and after zentorch compilation
        self.assertEqual(output_before, output_after)

        # Assert that the compilation pass has replaced Q/DQ ops
        self.assertTrue(
            expr=self.were_ops_replaced(quantized_model_compiled),
            msg="were_ops_replaced() not implemented or ops were not replaced "
                "by Zentorch compilation for 3D model.",
        )

    def quantize(self):
        quantized_model = torch.export.export_for_training(
            self.model,
            self.input,
        ).module()

        # Quantize model using PT2E X86InductorQuantizer
        x86_quantizer = X86InductorQuantizer()
        quant_config = Test_Pt2e_Quantization_Replacement().get_amd_x86_quantization_configuration()
        x86_quantizer.set_global(quant_config)

        quantized_model = prepare_pt2e(quantized_model, x86_quantizer)
        # Calibrate - run with example data
        quantized_model(*self.input)

        quantized_model = convert_pt2e(
            quantized_model,
            use_reference_representation=False,
            fold_quantize=True,
        )
        return quantized_model

    def were_ops_replaced(self, quantized_model):
        qdq_ops = {"quantize_per_tensor", "dequantize_per_tensor"}

        # Check for the presence of Q/DQ ops
        return all(
            not (
                node.op == 'call_function' and node.target in qdq_ops
            )
            for node in quantized_model.graph.nodes
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_XNNPack_Per_Channel_Quantization_Replacement(
    Simple_Quantized_Model_Test
):
    def test_xnnpack_quantization_replacement_flow(self):
        self.execute_test()

    def quantize(self):
        # Export model for quantization
        quantized_model = torch.export.export_for_training(
            self.model,
            self.input,
        ).module()

        # Quantize model using XNNPACKQuantizer
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        )

        quantized_model = prepare_pt2e(quantized_model, quantizer)
        quantized_model = convert_pt2e(
            quantized_model,
            use_reference_representation=False,
            fold_quantize=True,
        )
        return quantized_model

    def were_ops_replaced(self, quantized_model):
        qdq_ops = {"quantize_per_tensor", "dequantize_per_tensor"}

        # Check for the presence of Q/DQ ops
        return all(
            not (
                node.op == 'call_function' and node.target in qdq_ops
            )
            for node in quantized_model.graph.nodes
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_XNNPack_Quantization_Replacement(Simple_Quantized_Model_Test):
    def test_xnnpack_quantization_replacement_flow(self):
        self.execute_test()

    def quantize(self):
        # Export model for quantization
        quantized_model = torch.export.export_for_training(
            self.model,
            self.input,
        ).module()

        # Quantize model using XNNPACKQuantizer
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        )

        quantized_model = prepare_pt2e(quantized_model, quantizer)
        quantized_model = convert_pt2e(
            quantized_model,
            use_reference_representation=False,
            fold_quantize=True,
        )
        return quantized_model

    def were_ops_replaced(self, quantized_model):
        qdq_ops = {"quantize_per_tensor", "dequantize_per_tensor"}

        # Check for the presence of Q/DQ ops
        return all(
            not (
                node.op == 'call_function' and node.target in qdq_ops
            )
            for node in quantized_model.graph.nodes
        )


if __name__ == "__main__":
    run_tests()
