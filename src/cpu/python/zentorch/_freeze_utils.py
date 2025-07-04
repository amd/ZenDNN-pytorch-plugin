# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import contextlib
from ._freezing import _freeze
from ._utils import is_version_compatible_import


@contextlib.contextmanager
def freezing_enabled():
    # read previous/default values of freeze, this works even if
    # the user has set the values manually as we restore to that value
    previous_freeze_config = torch._inductor.config.freezing
    # monkey patch pytorch freeze
    torch._inductor.config.freezing = True
    if is_version_compatible_import(["_inductor", "freezing"], ["_freeze"]):
        # PT 2.7 or above
        previous_freeze_path = torch._inductor.freezing._freeze
        torch._inductor.freezing._freeze = _freeze
    else:
        # PT 2.6
        # TODO: remove this else block when dropping support for PT 2.6
        previous_freeze_path = torch._inductor.freezing.freeze
        torch._inductor.freezing.freeze = _freeze
    yield
    if is_version_compatible_import(["_inductor", "freezing"], ["_freeze"]):
        torch._inductor.freezing._freeze = previous_freeze_path
    else:
        # TODO: remove this else block when dropping support for PT 2.6
        torch._inductor.freezing.freeze = previous_freeze_path
    # reset to the previous values
    torch._inductor.config.freezing = previous_freeze_config
