# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from transformers import BertTokenizer
import random
import copy
import numpy as np
import torch
from torch._inductor import config
from packaging.version import parse
from torch.testing._internal.common_utils import TestCase, run_tests, SEED  # noqa: F401

try:
    import zentorch

    # for pattern matcher
    from zentorch._utils import counters

    has_zentorch = True
except ImportError:
    zentorch = None
    counters = None
    has_zentorch = False

supported_dtypes = [("float32")]
supported_dtypes_def = []
qlinear_dtypes = []
freeze_opt = [True, False]
freeze_def_opt = [False]
woq_dtypes = []


class DataTypes:

    mapDtypes = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int": torch.int,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }

    @classmethod
    def get_torch_type(cls, dtype: str):
        key = dtype.lower().strip()
        if key in cls.mapDtypes:
            return cls.mapDtypes[key]
        else:
            raise Exception("Unsupported DataType {dtype}")


class Range:
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max


B_RANGE = Range(1, 10)
M_RANGE = Range(2, 10)
K_RANGE = Range(2, 10)
N_RANGE = Range(2, 10)

P_RANGE = Range(1, 11)
Q_RANGE = Range(1, 11)
MATRIX_DIM_1_RANGE = Range(60, 60)
MATRIX_DIM_2_RANGE = Range(40, 40)
MATRIX_DIM_3_RANGE = Range(30, 30)
MATRIX_DIM_4_RANGE = Range(50, 50)

CONV_BS_RANGE = Range(1, 1)  # batch size
CONV_C_RANGE = Range(3, 3)  # number of channels
CONV_H_RANGE = Range(64, 64)  # height
CONV_WD_RANGE = Range(64, 64)  # width
CONV_OC_RANGE = Range(16, 16)  # output channels
CONV_KH_RANGE = Range(3, 3)  # kernel height
CONV_KW_RANGE = Range(3, 3)  # kernel width
CONV_DILATION2 = [[2, 2]]

EMB_R_RANGE = Range(11, 20)
EMB_W_RANGE = Range(1, 15)
EMB_D_RANGE = Range(2, 512)
EMB_MLP_OPT = [2]

MM_INPUT_SCALER_RANGE = Range(100, 100)

WOQ_M_RANGE = Range(1, 10)
WOQ_X_RANGE = Range(1, 10)
WOQ_Y_RANGE = Range(1, 10)
WOQ_K_RANGE = Range(3, 10)
WOQ_QZEROS_NONZERO_DIM_RANGE = Range(15, 15)

MM_ADD_1D_M_RANGE = Range(148, 148)
MM_ADD_1D_K_RANGE = Range(384, 384)
MM_ADD_1D_N_RANGE = Range(54, 54)
MM_ADD_2D_M_RANGE = Range(256, 256)
MM_ADD_2D_K_RANGE = Range(32, 32)
MM_ADD_2D_N_RANGE = Range(512, 512)
MM_ADD_3D_M_RANGE = Range(256, 256)
MM_ADD_3D_K_RANGE = Range(32, 32)
MM_ADD_3D_N_RANGE = Range(512, 512)
MM_ADD_3D_P_RANGE = Range(4, 4)
MM_ADD_3D_Q_RANGE = Range(64, 64)

if has_zentorch and zentorch._C.is_bf16_supported():
    woq_dtypes.append("bfloat16")


# Zentorch qlinear_* ops currently only support Float32 data type.
if has_zentorch and zentorch._C.is_avx512_supported():
    qlinear_dtypes.append(("float32"))
else:
    print(
        "Warning: Skipping zentorch qlinear Testcases since they are not \
supported on this hardware"
    )


if has_zentorch and zentorch._C.is_bf16_supported():
    supported_dtypes.append(("bfloat16"))
else:
    print(
        "Warning: Skipping Bfloat16 Testcases since they \
are not supported on this hardware"
    )

include_last_offset_opt = [True, False]
INCLUDE_LAST_OFFSET_OPT_DEF = [False]
scale_grad_opt = [True, False]
SCALE_GRAD_OPT_DEF = [False]
mode_opt = [0, 1, 2]
MODE_OPT_DEF = [0]
sparse_opt = [True, False]
SPARSE_OPT_DEF = [False]
input_dim_opt = [2, 3, 4]
INPUT_DIM_OPT_DEF = [2]
q_weight_list_opt = [0, 1]
Q_WEIGHT_LIST_OPT_DEF = [0]
bias_opt = [0, 1]
BIAS_OPT_DEF = [0]
woq_qzeros_opt = [0, 1]
group_size_opt = [-1, 1, 2, 3, 4, 5, 7, 8, 10]
group_size_def_opt = [-1]
q_granularity_opt = [
    "per_tensor",
    "per_channel",
]
Q_GRANULARITY_OPT_DEF = [
    "per_tensor"
]
q_zero_points_dtype_opt = [
    "int8",
    "uint8",
]
Q_ZERO_POINTS_DTYPE_OPT_DEF = [
    "int8"
]
q_linear_dtype_opt = [
    "float32",
    "bfloat16",
    "int8",
    "uint8",
]
Q_LINEAR_DTYPE_OPT_DEF = ["float32"]
conv_stride = [[1, 1], [2, 2]]
conv_stride_def = [[1, 1]]
conv_padding = [[0, 0], [1, 1]]
conv_padding_def = [[0, 0]]
seq_length_opt = [384, 512]
batch_size_opt = [1, 4, 8]
mask_type_opt = ["none", "float", "bfloat16", "bool"]
num_heads_opt = [12, 16]
head_dim_opt = [32, 64]

at_ops = torch.ops.aten
zt_ops = torch.ops.zentorch

qlinear_eltwise_map = {
    "relu": (torch.nn.ReLU(), zt_ops.zentorch_qlinear_relu.default),
    "sigmoid": (torch.nn.Sigmoid(), zt_ops.zentorch_qlinear_sigmoid.default),
    # TODO: Enable once silu, gelu_erf, gelu_tanh fusions are supported
    # with qlinear
    # "silu": (torch.nn.SiLU(), zt_ops.zentorch_qlinear_silu.default),
    # "gelu_erf": (torch.nn.GELU(), zt_ops.zentorch_qlinear_gelu_erf.default),
    # "gelu_tanh": (
    #     torch.nn.GELU(approximate="tanh"),
    #     zt_ops.zentorch_qlinear_gelu_tanh.default,
    # ),
}
QLINEAR_ELTWISE_OPT_DEF = ["relu"]


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


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
    torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache.clear()


# Method to hadle test with freezeing enable
# and parameterized based on freezing option
def test_with_freeze_opt(compiled_graph, inputs, freeze_opt):
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    config.freezing = freeze_opt
    return compiled_graph(*inputs)


# Singleton class definition
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Base test case class with common methods
class BaseZentorchTestCase(TestCase):
    def setUp(self):
        set_seed()

    def skip_if_bfloat16_path_issue(self, dtype):
        if dtype == "bfloat16":
            self.skipTest("Skipping it due to issue with BF16 path.")

    def skip_if_bfloat16_unsupported_hardware(self):
        if not zentorch._C.is_bf16_supported():
            self.skipTest(
                "Warning: Skipping Bfloat16 Testcases since they are not "
                + "supported on this hardware"
            )

    def skip_if_bfloat16_not_yet_supported(self, dtype):
        if dtype == "bfloat16":
            self.skipTest(
                "Warning: Skipping Bfloat16 Testcases since they are not "
                + "yet supported"
            )


print("Seed is", SEED)
set_seed(SEED)

skip_test_pt_2_0 = False
skip_test_pt_2_1 = False
skip_test_pt_2_3 = False
skip_test_pt_2_4 = False

# Get the current version of torch
torch_version = torch.__version__

# Parse the version
parsed_version = parse(torch_version)
if parsed_version.major == 2 and parsed_version.minor == 0:
    skip_test_pt_2_0 = True

if parsed_version.major == 2 and parsed_version.minor == 1:
    skip_test_pt_2_1 = True

if parsed_version.major == 2 and parsed_version.minor < 3:
    skip_test_pt_2_3 = True

if parsed_version.major == 2 and parsed_version.minor < 4:
    skip_test_pt_2_4 = True


# Unified Test_Data class
class Test_Data(metaclass=Singleton):
    def __init__(self):
        # Define the dtypes attribute
        self.dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "int": torch.int,
            "int8": torch.int8,
            "uint8": torch.uint8,
        }

    def create_llm_data(self, dtype="float32"):
        # Add data if required in llm_tests
        pass

    def create_pretrained_model_data(
        self, dtype="float32", model_name="bert-large-uncased"
    ):
        batch_size = random.randint(1, 100)
        # torch.rand() is not supported for integer
        # since we have some int tests that calls create_pretrained_model_data with
        # dtype torch.int - we need to handle the creation
        # of input_scalar for that case
        self.input3d = torch.randn(batch_size, 3, 224, 224).type(self.dtypes[dtype])
        input_text = "This is a sample input sentence for testing Bert Model."
        tokenizer = BertTokenizer.from_pretrained(model_name)
        input_ids = tokenizer.encode(input_text, add_special_tokens=True)
        self.input_tensor = torch.tensor(input_ids).unsqueeze(0)

    def create_unittest_data(self, dtype="float32", group_size=-1):
        torch_type = self.get_torch_type(dtype)
        self.b, self.m, self.k, self.n = (
            torch.randint(1, 11, (1,)).item() for _ in range(4)
        )

        # m*k, k*n, m*n
        self.x = torch.randn(self.m, self.k).type(torch_type)
        self.y = torch.randn(self.k, self.n).type(torch_type)
        self.result = torch.zeros(self.m, self.n).type(torch_type)

        self.input = torch.randn(self.m, self.n).type(torch_type)
        self.input1d = torch.randn(self.n).type(torch_type)
        # torch.rand() is not supported for integer
        # since we have some int tests that calls create_unittest_data with
        # dtype torch.int - we need to handle the creation
        # of input_scalar for that case
        if torch_type in [torch.bfloat16, torch.float32]:
            self.input_scalar = torch.rand(()).type(torch_type)
        else:
            self.input_scalar = torch.randint(0, 100, ()).type(torch_type)

        self.empty_bias = torch.zeros(0).type(torch_type)
        self.result_m = torch.zeros(int(self.m)).type(torch_type)
        self.result_1 = torch.zeros(1).type(torch_type)

        self.A = torch.randn(self.m, 1).type(torch_type)
        self.B = torch.randn(1, self.m).type(torch_type)

        # b*m*k, b*k*n, b*m*n
        self.x3d = torch.randn(self.b, self.m, self.k).type(torch_type)
        self.y3d = torch.randn(self.b, self.k, self.n).type(torch_type)
        self.input3d = torch.randn(self.b, self.m, self.n).type(torch_type)

        self.R = torch.randint(11, 20, (1,)).item()
        self.W = torch.randint(1, 15, (1,)).item()
        self.embedding_matrix = torch.randn(self.R, 3).type(torch_type)
        self.emb_input = torch.randint(0, self.R, (self.W,))
        self.offsets = torch.tensor([0, self.W])
        self.mlp_inputs = torch.randn(2, self.k)

        self.M = [
            torch.randn(60, 30).type(torch_type),
            torch.randn(30).type(torch_type),
        ]
        self.T1 = [torch.randn(2, 30, 30).type(torch_type)]
        self.x1 = [
            torch.randn(60, 40).type(torch_type),
            torch.randn(40, 60).transpose(0, 1).type(torch_type),
        ]

        self.y1 = [
            torch.randn(40, 30).type(torch_type),
            torch.randn(30, 40).transpose(1, 0).type(torch_type),
        ]

        self.M2 = torch.randn(60, 30, 50).type(torch_type)
        self.M3 = torch.randn(50).type(torch_type)

        self.x2 = [
            torch.randn(60, 30, 40).type(torch_type),
            torch.randn(60, 40, 30).transpose(1, 2).type(torch_type),
            torch.randn(30, 60, 40).transpose(0, 1).type(torch_type),
        ]

        self.y2 = [
            torch.randn(60, 40, 50).type(torch_type),
            torch.randn(60, 50, 40).transpose(1, 2).type(torch_type),
            torch.randn(50, 40, 60).transpose(0, 2).type(torch_type),
        ]

        # Test data for serialized zentorch_qlinear.
        self.y_int8_square = [
            torch.randint(-128, 127, (self.k, self.k)).type(torch.int8),
        ]
        self.bias_for_qlinear_square = [
            None,
            torch.randn(self.k).type(torch_type),
        ]
        self.y_scales_square = {
            "per_tensor": torch.randn((1,)).type(torch.float32),
            "per_channel": torch.randn(self.k).type(torch.float32),
        }
        self.y_zero_points_square = {
            "per_tensor": torch.tensor(0).type(torch.int8),
            "per_channel": torch.zeros(self.k).type(torch.int8),
        }
        self.p, self.q = (torch.randint(1, 11, (1,)).item() for _ in range(2))

        self.x_for_qlinear = {
            "float32": {
                2: torch.randn(self.m, self.k).type(torch.float32),
                3: torch.randn(self.m, self.p, self.k).type(torch.float32),
                4: torch.randn(self.m, self.p, self.q, self.k).type(torch.float32),
            },
            "bfloat16": {
                2: torch.randn(self.m, self.k).type(torch.bfloat16),
                3: torch.randn(self.m, self.p, self.k).type(torch.bfloat16),
                4: torch.randn(self.m, self.p, self.q, self.k).type(torch.bfloat16),
            },
            "int8": {
                2: torch.randint(-128, 127, (self.m, self.k)).type(torch.int8),
                3: torch.randint(-128, 127, (self.m, self.p, self.k)).type(torch.int8),
                4: torch.randint(-128, 127, (self.m, self.p, self.q, self.k)).type(
                    torch.int8
                ),
            },
            "uint8": {
                2: torch.randint(0, 255, (self.m, self.k)).type(torch.uint8),
                3: torch.randint(0, 255, (self.m, self.p, self.k)).type(torch.uint8),
                4: torch.randint(0, 255, (self.m, self.p, self.q, self.k)).type(
                    torch.uint8
                ),
            },
        }
        self.y_int8 = [
            torch.randint(-128, 127, (self.k, self.n)).type(torch.int8).t(),
            torch.randint(-128, 127, (self.n, self.k)).type(torch.int8),
        ]
        self.binary_input = {
            2: torch.randn(self.m, self.n),
            3: torch.randn(self.m, self.p, self.n),
            4: torch.randn(self.m, self.p, self.q, self.n),
        }
        self.bias_for_qlinear = [
            None,
            torch.randn(self.n).type(torch_type),
        ]
        self.x_scales = {
            "per_tensor": torch.randn((1,)).type(torch.float32),
        }
        self.x_zero_points = {
            "per_tensor": {
                "float32": {
                    # Scalar Tensor
                    "int8": torch.tensor(0).type(torch.int8),
                    # 1D Tensor
                    "uint8": torch.randint(0, 255, (1,)).type(torch.uint8),
                },
                "bfloat16": {
                    # Scalar Tensor
                    "int8": torch.tensor(0).type(torch.int8),
                    # 1D Tensor
                    "uint8": torch.randint(0, 255, (1,)).type(torch.uint8),
                },
                "int8": {
                    # 1D Tensor
                    "int8": torch.zeros(1).type(torch.int8),
                    # Scalar Tensor
                    "uint8": torch.tensor(0).type(
                        torch.int8
                    ),  # made it int8 as uint8 is not supported
                },
                "uint8": {
                    "int8": torch.randint(0, 255, (1,)).type(
                        torch.uint8
                    ),  # made it uint8 as int8 is not supported
                    "uint8": torch.randint(0, 255, (1,)).type(torch.uint8),
                },
            },
        }
        self.y_scales = {
            "per_tensor": torch.randn((1,)).type(torch.float32),
            "per_channel": torch.randn(self.n).type(torch.float32),
        }
        self.y_zero_points = {
            "per_tensor": torch.tensor(0).type(torch.int8),
            "per_channel": torch.zeros(self.n).type(torch.int8),
        }
        self.output_scales = {
            "per_tensor": {
                "float32": {
                    "positive_scales": None,
                    # "negative_scales" : None,
                },
                "bfloat16": {
                    "positive_scales": None,
                    # "negative_scales" : None,
                },
                "uint8": {
                    "positive_scales": torch.rand((1,)).type(torch.float32),
                    # "negative_scales" : torch.rand((1,)).type(torch.float32) * -1,
                },
                "int8": {
                    "positive_scales": torch.rand((1,)).type(torch.float32),
                    # "negative_scales" : torch.rand((1,)).type(torch.float32) * -1,
                },
            }
        }
        self.output_zero_points = {
            "per_tensor": {
                "float32": None,
                "bfloat16": None,
                "uint8": torch.randint(0, 255, (1,)).type(torch.uint8),
                "int8": torch.zeros(1).type(torch.int8),
            },
        }
        self.wrong_scales_per_channel = torch.randn(self.n + 1).type(torch.float32)
        self.wrong_zero_points_per_channel = torch.zeros(self.n + 1).type(torch.int8)

        self.woq_m, self.woq_x, self.woq_y = (
            torch.randint(1, 11, (1,)).item() for _ in range(3)
        )
        self.woq_group_size = group_size
        self.packing_ratio = 8
        if group_size == -1:
            self.woq_k = torch.randint(3, 11, (1,)).item() * self.packing_ratio
            group_size = self.woq_k
        else:
            self.woq_k = (
                torch.randint(3, 11, (1,)).item() * self.packing_ratio * group_size
            )
        # This is done for supporting the add/mul operation in unit test.
        self.woq_n = self.woq_k

        self.woq_input = {
            2: torch.randn(self.woq_m, self.woq_k).type(torch_type),
            3: torch.randn(self.woq_m, self.woq_y, self.woq_k).type(torch_type),
            4: torch.randn(self.woq_m, self.woq_x, self.woq_y, self.woq_k).type(
                torch_type
            ),
        }
        self.woq_add = {
            2: torch.randn(self.woq_m, self.woq_n).type(torch_type),
            3: torch.randn(self.woq_m, self.woq_y, self.woq_n).type(torch_type),
            4: torch.randn(self.woq_m, self.woq_x, self.woq_y, self.woq_n).type(
                torch_type
            ),
        }
        self.woq_mul = {
            2: torch.randn(self.woq_m, self.woq_n).type(torch_type),
            3: torch.randn(self.woq_m, self.woq_y, self.woq_n).type(torch_type),
            4: torch.randn(self.woq_m, self.woq_x, self.woq_y, self.woq_n).type(
                torch_type
            ),
        }
        woq_qweight = torch.randn(self.woq_k, self.woq_n // self.packing_ratio).type(
            torch.int32
        )
        # Here we are creating two different data copies to deal with weight caching
        # issue.
        self.woq_qweight = {
            "bfloat16": copy.deepcopy(woq_qweight),
            "float32": copy.deepcopy(woq_qweight),
        }
        woq_scales = torch.randn(self.woq_k // group_size, self.woq_n).type(
            torch.bfloat16
        )
        self.woq_scales = {
            "bfloat16": copy.deepcopy(woq_scales),
            "float32": copy.deepcopy(woq_scales.type(torch.float32)),
        }
        self.woq_qzeros = [
            None,
            torch.zeros(
                self.woq_k // group_size, self.woq_n // self.packing_ratio
            ).type(torch.int32),
        ]
        self.woq_qzeros_nonzero = torch.randint(
            1, 15, (self.woq_k // group_size, self.woq_n // self.packing_ratio)
        ).type(torch.int32)
        self.woq_bias = [
            None,
            torch.randn(self.woq_n).type(torch_type),
        ]
        self.conv_input = (
            torch.randn(1, 3, 64, 64)
            .type(torch_type)
            .to(memory_format=torch.channels_last)
        )
        self.conv_weight = (
            torch.randn(16, 3, 3, 3)
            .type(torch_type)
            .to(memory_format=torch.channels_last)
        )
        self.conv_bias = torch.randn(16).type(torch_type)
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.dilation = [1, 1]
        self.output_padding = [0, 0]
        self.conv_input3d = torch.randn(1, 3, 3).type(torch_type)
        self.conv_weight3d = torch.randn(1, 3, 3).type(torch_type)
        self.dilation2 = [2, 2]

        self.mm_add_1D = [
            torch.rand(148, 384).type(torch_type),
            torch.rand(384, 54).type(torch_type),
            torch.rand(54).type(torch_type),
        ]

        self.mm_add_2D = [
            torch.rand(256, 32).type(torch_type),
            torch.rand(32, 512).type(torch_type),
            torch.rand(256, 512).type(torch_type),
        ]

        self.mm_add_3D = [
            torch.rand(256, 32).type(torch_type),
            torch.rand(32, 512).type(torch_type),
            torch.rand(4, 64, 512).type(torch_type),
        ]

    # Create data for addmm tests
    # Ensure data creation for this cateogry tests in this function
    def create_data_addmm(
        self,
        dtype,
        b,
        m,
        k,
        n,
        M,
        T1,
        x1,
        y1,
        M2,
        M3,
        x2,
        y2,
        x,
        y,
        x3d,
        y3d,
        input,
        input3d,
        input1d,
    ):
        self.b, self.m, self.k, self.n = (b, m, k, n)
        self.M = M
        self.T1 = T1
        self.x1 = x1
        self.y1 = y1
        self.M2 = M2
        self.M3 = M3
        self.x2 = x2
        self.y2 = y2
        self.x = x
        self.y = y
        self.x3d = x3d
        self.y3d = y3d
        self.input = input
        self.input1d = input1d
        self.input3d = input3d

    # Create data for convolution tests
    # Ensure data creation for this cateogry tests in this function
    def create_data_conv(
        self,
        dtype,
        conv_input,
        conv_weight,
        conv_bias,
        stride,
        padding,
        dilation,
        output_padding,
        conv_input3d,
        conv_weight3d,
        dilation2,
    ):
        # Create data for convolution
        self.conv_input = conv_input
        self.conv_weight = conv_weight
        self.conv_bias = conv_bias

        # Set default values for stride, padding, dilation, and output_padding
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding

        # Create 3D convolution data
        self.conv_input3d = conv_input3d
        self.conv_weight3d = conv_weight3d
        self.dilation2 = dilation2

    # Create data for embedding tests
    # Ensure data creation for this cateogry tests in this function
    def create_data_emb(
        self,
        dtype,
        R,
        W,
        k,
        embedding_matrix,
        emb_input,
        offsets,
        mlp_inputs,
    ):
        self.R = R
        self.W = W
        self.k = k
        self.embedding_matrix = embedding_matrix
        self.emb_input = emb_input
        self.offsets = offsets
        self.mlp_inputs = mlp_inputs

    # Create data for mm tests
    # Ensure data creation for this cateogry tests in this function
    def create_data_mm(
        self,
        dtype,
        b,
        m,
        k,
        n,
        x,
        y,
        result,
        input,
        input1d,
        input_scalar,
        empty_bias,
        result_m,
        result_1,
        A,
        B,
        x3d,
        y3d,
        input3d,
    ):
        self.b, self.m, self.k, self.n = (b, m, k, n)
        # m*k, k*n, m*n
        self.x = x
        self.y = y
        self.result = result

        self.input = input
        self.input1d = input1d

        self.input_scalar = input_scalar

        self.empty_bias = empty_bias
        self.result_m = result_m
        self.result_1 = result_1

        self.A = A
        self.B = B

        # b*m*k, b*k*n, b*m*n
        self.x3d = x3d
        self.y3d = y3d
        self.input3d = input3d

    # Create data for woq_linear tests
    # Ensure data creation for this cateogry tests in this function
    def create_data_woq(
        self,
        dtype,
        woq_m,
        woq_x,
        woq_y,
        woq_k,
        b,
        m,
        n,
        packing_ratio,
        group_size_val,
        woq_input,
        woq_add,
        woq_mul,
        woq_qweight,
        woq_scales,
        woq_qzeros,
        woq_qzeros_nonzero,
        woq_bias,
        input3d,
        input1d
    ):
        self.woq_m, self.woq_x, self.woq_y, self.woq_k = (
            woq_m, woq_x, woq_y, woq_k
        )
        self.b, self.m, self.n = (
            b, m, n
        )

        self.woq_group_size = group_size_val
        self.packing_ratio = packing_ratio

        self.woq_input = woq_input
        self.woq_add = woq_add
        self.woq_mul = woq_mul
        woq_qweight = woq_qweight

        self.woq_qweight = woq_qweight
        woq_scales = woq_scales
        self.woq_scales = woq_scales
        self.woq_qzeros = woq_qzeros
        self.woq_qzeros_nonzero = woq_qzeros_nonzero
        self.woq_bias = woq_bias
        self.input3d = input3d
        self.input1d = input1d

    # Create data for qlinear tests
    # Ensure data creation for this cateogry tests in this function
    def create_data_qlinear(
        self,
        dtype,
        b,
        m,
        p,
        q,
        k,
        n,
        y_int8_square,
        bias_for_qlinear_square,
        y_scales_square,
        y_zero_points_square,
        x_for_qlinear,
        y_int8,
        binary_input,
        bias_for_qlinear,
        x_scales,
        x_zero_points,
        y_scales,
        y_zero_points,
        output_scales,
        output_zero_points,
        wrong_scales_per_channel,
        wrong_zero_points_per_channel,
        y,
        input1d,
        x1,
        y1,
        x3d,
        y3d,
        input3d,
    ):
        self.b, self.m, self.p, self.q, self.k, self.n = (b, m, p, q, k, n)
        self.y_int8_square = y_int8_square
        self.bias_for_qlinear_square = bias_for_qlinear_square
        self.y_scales_square = y_scales_square
        self.y_zero_points_square = y_zero_points_square
        self.x_for_qlinear = x_for_qlinear
        self.y_int8 = y_int8
        self.binary_input = binary_input
        self.bias_for_qlinear = bias_for_qlinear
        self.x_scales = x_scales
        self.x_zero_points = x_zero_points
        self.y_scales = y_scales
        self.y_zero_points = y_zero_points
        self.output_scales = output_scales
        self.output_zero_points = output_zero_points
        self.wrong_scales_per_channel = wrong_scales_per_channel
        self.wrong_zero_points_per_channel = wrong_zero_points_per_channel
        self.y = y
        self.input1d = input1d
        self.x1 = x1
        self.y1 = y1
        self.x3d = x3d
        self.y3d = y3d
        self.input3d = input3d

    # Create data for tests using mm_add_1D, mm_add_2D and mm_add_3D
    # Used in test_addmm.py
    def create_data_mm_add_xD(
        self,
        dtype,
        mm_add_1D,
        mm_add_2D,
        mm_add_3D,
    ):
        self.mm_add_1D = mm_add_1D
        self.mm_add_2D = mm_add_2D
        self.mm_add_3D = mm_add_3D

    # TODO ZENAI-1522
    # Change str_type -> str_type.lower()
    def get_torch_type(self, str_type):
        return self.dtypes[str_type.lower()]
