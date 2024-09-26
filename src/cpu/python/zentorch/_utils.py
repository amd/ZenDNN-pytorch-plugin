# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from torch.fx import passes
from os import environ
import collections

counters = collections.defaultdict(collections.Counter)


def save_graph(fx_graph, graph_name):
    env_var = "ZENTORCH_SAVE_GRAPH"
    if env_var in environ and environ[env_var] == "1":
        g = passes.graph_drawer.FxGraphDrawer(fx_graph, graph_name)
        with open(f"{graph_name}.svg", "wb") as f:
            f.write(g.get_dot_graph().create_svg())


def add_version_suffix(major: str, minor: str, patch: str = 0):
    # This function will add a ".dev" substring to the input arguments.
    # This will extend the pytorch version comparisions done using TorchVersion
    # class to include nightly and custom build versions as well.
    # The following tables shows the behaviour of TorchVersion comparisons
    # for release, nightlies and custom binaries, when the substring is used.
    # ".dev" is added to second column i.e A.B.C -> A.B.C.dev

    # This function is intended for only lesser than comparisons.

    #                           X.Y.Z < A.B.C
    # +---------------+----------------+-----------------+
    # | Torch Version |  Torch Version |  Implementation |
    # | used by user  |      to be     |    Behaviour    |
    # |    (X.Y.Z)    |  compared with |                 |
    # |               |    (A.B.C)     |                 |
    # +---------------+----------------+-----------------+
    # |      2.3.1    |      2.4.0     |      True       |
    # +---------------+----------------+-----------------+
    # |      2.4.0    |      2.4.0     |      False      |
    # +---------------+----------------+-----------------+
    # |    2.4.0.dev  |      2.4.0     |      False      |
    # |   (Nightly    |                |                 |
    # |    binaries)  |                |                 |
    # +---------------+----------------+-----------------+
    # |    2.5.0.dev  |      2.4.0     |      False      |
    # |    2.6.0.dev  |                |                 |
    # |   (Nightly    |                |                 |
    # |    binaries)  |                |                 |
    # +---------------+----------------+-----------------+
    # |  2.4.0a0+git  |    2.4.0       |      False      |
    # |    d990dad    |                |                 |
    # +---------------+----------------+-----------------+

    return f"{major}.{minor}.{patch}.dev"
