# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************


import torch  # noqa: F401
from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files


from ._prepack_pass import add_zentorch_weight_prepack_ops
from ._op_replacements_new import replace_with_zentorch_ops_new
from ._unary_fusions import zentorch_unary_post_op_fusions
from ._unary_binary_fusions import zentorch_unary_binary_post_op_fusions
from ._binary_binary_fusions import zentorch_binary_binary_post_op_fusions


# we need another optimize function for export path as all the current passes
# in the original optimize function are not supported by proxy executor
# TODO: align this optimize with the one for compile path once the shim
# files are added and all issues are resolved. When aligned, then the below
# function can be removed and we will reuse the old optimize in export as well.
def optimize_for_export(fx_graph):
    fx_graph = replace_with_zentorch_ops_new(fx_graph)
    if config.freezing:
        fx_graph = add_zentorch_weight_prepack_ops(fx_graph)
    fx_graph = zentorch_binary_binary_post_op_fusions(fx_graph)
    fx_graph = zentorch_unary_binary_post_op_fusions(fx_graph)
    fx_graph = zentorch_unary_post_op_fusions(fx_graph)
    return fx_graph


class OptimizePassExport(CustomGraphPass):
    def __call__(self, graph: torch.fx.Graph):
        optimize_for_export(graph)

    def uuid(self):
        # needed for inductor caching
        return get_hash_for_files((__file__,))


export_optimize_pass = OptimizePassExport()
