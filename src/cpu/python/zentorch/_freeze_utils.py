# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import torch
import contextlib
from deprecated import deprecated


@deprecated(
    "zentorch.freezing_enabled() context is deprecated and will be removed in next release."
    " Kindly use the env TORCHINDUCTOR_FREEZING to enable or disable freezing."
)
@contextlib.contextmanager
def freezing_enabled():
    # read previous/default values of freeze, this works even if
    # the user has set the values manually as we restore to that value
    previous_freeze_config = torch._inductor.config.freezing
    torch._inductor.config.freezing = True
    yield
    # reset to the previous values
    torch._inductor.config.freezing = previous_freeze_config
