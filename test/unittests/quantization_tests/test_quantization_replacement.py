# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

try:  # noqa: SIM105
    import quark  # noqa: F401
    import zentorch
    from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize  # noqa: F401

except ImportError:
    pass

import unittest
import torch
from torch.fx import Graph, GraphModule
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
)


def create_quark_quantization_fx_graph():
    # Create a new graph
    graph = Graph()

    # Add placeholders for all the inputs with proper metadata
    primals_1 = graph.placeholder('primals_1')  # input_scale
    primals_1.meta['val'] = torch.tensor(1.0, dtype=torch.float32)

    primals_2 = graph.placeholder('primals_2')  # input_zero_point
    primals_2.meta['val'] = torch.tensor(0, dtype=torch.int32)

    primals_3 = graph.placeholder('primals_3')  # input_tensor
    primals_3.meta['val'] = torch.randn(2, 8, dtype=torch.float32)

    primals_4 = graph.placeholder('primals_4')  # weight_tensor
    primals_4.meta['val'] = torch.randn(4, 8, dtype=torch.float32)

    primals_5 = graph.placeholder('primals_5')  # weight_scale
    primals_5.meta['val'] = torch.randn(4, dtype=torch.float32)

    primals_6 = graph.placeholder('primals_6')  # weight_zero_point
    primals_6.meta['val'] = torch.zeros(4, dtype=torch.int32)

    primals_7 = graph.placeholder('primals_7')  # bias
    primals_7.meta['val'] = torch.randn(4, dtype=torch.float32)

    # Create the input quantization node
    scaled_fake_quantize = graph.call_function(
        torch.ops.quark.scaled_fake_quantize.default,
        args=(
            'uint8',     # quant_dtype
            primals_3,   # input_tensor
            primals_1,   # input_scale
            primals_2,   # input_zero_point
            1,           # axis
            1,           # group_size
            0.0,         # quant_min
            255.0,       # quant_max
            8,           # round_mode
            'per_tensor',  # qscheme
            'None'       # mx_element_dtype
        )
    )
    scaled_fake_quantize.meta['val'] = torch.randn(2, 8, dtype=torch.float32)

    # Create the weight quantization node
    scaled_fake_quantize_1 = graph.call_function(
        torch.ops.quark.scaled_fake_quantize.default,
        args=(
            'int8',       # quant_dtype
            primals_4,    # weight_tensor
            primals_5,    # weight_scale
            primals_6,    # weight_zero_point
            0,            # axis
            1,            # group_size
            -128.0,       # quant_min
            127.0,        # quant_max
            8,            # round_mode
            'per_channel',  # qscheme
            'None'        # mx_element_dtype
        )
    )
    scaled_fake_quantize_1.meta['val'] = torch.randn(4, 8, dtype=torch.float32)

    # Create the permute node
    permute = graph.call_function(
        torch.ops.aten.permute.default,
        args=(scaled_fake_quantize_1, [1, 0])
    )
    permute.meta['val'] = torch.randn(8, 4, dtype=torch.float32)

    # Create the zentorch_addmm_1dbias node (the root of our pattern)
    zentorch_addmm_1dbias = graph.call_function(
        torch.ops.zentorch.zentorch_addmm_1dbias.default,
        args=(primals_7, scaled_fake_quantize, permute)
    )
    zentorch_addmm_1dbias.meta['val'] = torch.randn(2, 4, dtype=torch.float32)

    # Create the output node
    graph.output(zentorch_addmm_1dbias)

    # Create a dummy module to wrap the graph
    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

    module = DummyModule()

    # Create the GraphModule
    gm = GraphModule(module, graph)

    return gm


@unittest.skipIf(
    condition="quark" not in sys.modules or not has_zentorch,
    reason="Either Quark or ZENTORCH is not installed",
)
class Test_Quark_Pattern_Replacement_Procedural_Graph(Zentorch_TestCase):
    def test_quark_pattern_replacement_procedural(self):
        # Create the FX graph that matches our pattern
        gm = create_quark_quantization_fx_graph()

        # Apply the replacement
        replaced_gm = zentorch._qop_replacement.replace_with_zentorch_qops(gm)

        # Check if replacement occurred
        quark_ops_found = any(
            node.op == 'call_function'
            and hasattr(node.target, 'name')
            and 'quark' in str(node.target)
            for node in replaced_gm.graph.nodes
        )

        zentorch_qlinear_found = any(
            node.op == 'call_function'
            and hasattr(node.target, 'name')
            and 'zentorch_qlinear' in str(node.target)
            for node in replaced_gm.graph.nodes
        )

        # Assert that replacement worked
        self.assertFalse(quark_ops_found, "Quark ops should be replaced")
        self.assertTrue(zentorch_qlinear_found, "ZenTorch qlinear should be present")


if __name__ == "__main__":
    run_tests()
