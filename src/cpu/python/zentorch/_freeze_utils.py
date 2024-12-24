# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import contextlib
from ._freezing import freeze


@contextlib.contextmanager
def freezing_enabled():
    # read previous/default values of freeze, this works even if
    # the user has set the values manually as we restore to that value
    previous_freeze_config = torch._inductor.config.freezing
    previous_freeze_path = torch._inductor.freezing.freeze
    # monkey patch pytorch freeze
    torch._inductor.config.freezing = True
    torch._inductor.freezing.freeze = freeze
    yield
    # reset to the previous values
    torch._inductor.config.freezing = previous_freeze_config
    torch._inductor.freezing.freeze = previous_freeze_path
