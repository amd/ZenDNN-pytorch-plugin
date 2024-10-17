# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import random
import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase, run_tests, SEED  # noqa: F401

supported_dtypes = [("float32")]

try:
    import zentorch

    # for pattern matcher
    from zentorch._utils import counters

    has_zentorch = True
except ImportError:
    zentorch = None
    counters = None
    has_zentorch = False


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


if has_zentorch and zentorch._C.is_bf16_supported():
    supported_dtypes.append(("bfloat16"))
else:
    print(
        "Warning: Skipping Bfloat16 Testcases since they \
are not supported on this hardware"
    )
print("Seed is", SEED)
set_seed(SEED)

skip_test_pt_2_0 = False
skip_test_pt_2_3 = False

if torch.__version__[:3] == "2.0":
    skip_test_pt_2_0 = True

if torch.__version__[:3] < "2.3":
    skip_test_pt_2_3 = True


# when calling the torch.compile flow, we need inference_mode decorator
# that is not needed when invoking zentorch ops directly
def reset_dynamo():
    # if TorchVersion(torch.__version__) < "2.3":
    #     torch._dynamo.reset()
    # Though dynamo reset is not needed for switching between backends
    # it will still help us in clearing the cache
    # if cache limit has reached new compile backends
    # wouldn't be pass through zentorch.optimize
    # WARNING: torch._dynamo hit config.cache_size_limit (8)
    torch._dynamo.reset()
