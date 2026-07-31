# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
from torch.testing import FileCheck
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    EmbTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    update_supported_dtypes,
    freeze_opt,
    test_with_freeze_opt,
    test_with_freeze_opt_and_cpp_wrapper,
    counters,
)

supported_dtypes = update_supported_dtypes(supported_dtypes, "zentorch_embedding")


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Embedding(nn.Module):
    def __init__(self, embedding_dim, dtype=torch.float):
        super(Custom_Model_Embedding, self).__init__()
        self.embedding = nn.Embedding(10000, embedding_dim, dtype=dtype)

    def forward(self, input):
        embed = self.embedding(input)
        return embed


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Embedding_Model(EmbTestCase):
    @EmbTestCase.hypothesis_params_emb_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_embedding_compile_model(self, dtype, freeze_opt):
        new_dtype = self.data.get_torch_type(dtype)
        model = Custom_Model_Embedding(256, dtype=new_dtype)
        input = torch.randint(0, 10000, (10,))
        model_output = model(input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_embedding"], 0)
        compiled_graph_output = test_with_freeze_opt(
            compiled_graph, (input), freeze_opt
        )
        self.assertEqual(counters["zentorch"]["zentorch_embedding"], 1)
        self.assertEqual(model_output, compiled_graph_output)


class _EmbeddingModule(torch.nn.Module):
    """A plain ``nn.Embedding`` model.

    torch.compile places an ``aten.embedding`` in the graph. Under
    ``backend="zentorch"`` it is replaced by ``zentorch_embedding`` and, with
    ``cpp_wrapper``, lowered through the ``aoti_torch_cpu_zentorch_embedding``
    shim + FallbackKernel. The eager forward runs the same nn.Embedding and
    serves as the numerical reference. The weight is kept trainable
    (``freeze=False``) so the op is not const-folded away before the zentorch
    replacement runs.
    """

    def __init__(self, weight, padding_idx, scale_grad_by_freq, sparse):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            weight,
            freeze=False,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

    def forward(self, indices):
        return self.embedding(indices)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Embedding_Shim_Model(EmbTestCase):
    """AOTI test for the embedding op, following PyTorch's two pillars:
      Pillar 1 (numerical): the backend='zentorch' output (where aten.embedding
                            is replaced by zentorch_embedding) matches the eager
                            reference.
      Pillar 2 (codegen):   under cpp_wrapper the op lowers through the
                            aoti_torch_cpu_zentorch_embedding shim.
    A Hypothesis test sweeps the (dtype x sparse x scale_grad x freeze x
    cpp_wrapper) combinations."""

    @EmbTestCase.hypothesis_params_emb_itr(
        dtype_list=supported_dtypes,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_embedding_shim_model(
        self, dtype, sprs_opt, scale_opt, freeze_opt, cpp_wrapper
    ):
        padding_idx = -1
        # A real nn.Embedding(padding_idx=k) zeros weight[k] at construction;
        # from_pretrained does NOT. Mirror that invariant so the padding row is
        # zero -- otherwise zentorch_embedding (which zeros the padding row in
        # the forward output regardless of the weight value) would diverge from
        # eager on any index that lands on the padding row.
        weight = self.data.embedding_matrix.clone()
        weight[padding_idx] = 0
        model = _EmbeddingModule(weight, padding_idx, scale_opt, sprs_opt).eval()
        indices = self.data.emb_input

        # Pillar 1 reference: eager, the PyTorch oracle (uniform with the other
        # AOTI tests). Captured before compilation.
        eager_out = model(indices)

        reset_dynamo()
        zentorch_graph = torch.compile(model, backend="zentorch")
        zentorch_out, cpp_code = test_with_freeze_opt_and_cpp_wrapper(
            zentorch_graph, (indices,), freeze_opt, cpp_wrapper
        )

        # Pillar 1: numerical equivalence vs eager.
        self.assertEqual(zentorch_out.dtype, eager_out.dtype)
        self.assertEqual(zentorch_out, eager_out, atol=1e-3, rtol=1e-3)

        # Pillar 2 (codegen): op lowers to its AOTI C-shim (see helper docstring).
        if cpp_wrapper:
            FileCheck().check("aoti_torch_cpu_zentorch").run(cpp_code)


if __name__ == "__main__":
    run_tests()
