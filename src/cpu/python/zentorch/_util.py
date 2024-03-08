# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from torch.fx import passes
from os import environ


def save_graph(fx_graph, graph_name):
    env_var = "ZENTORCH_SAVE_GRAPH"
    if env_var in environ and environ[env_var] == "1":
        g = passes.graph_drawer.FxGraphDrawer(fx_graph, graph_name)
        with open(f"{graph_name}.svg", "wb") as f:
            f.write(g.get_dot_graph().create_svg())
