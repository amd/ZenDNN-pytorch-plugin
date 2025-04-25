# ******************************************************************************
# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Was sourced from
# https://github.com/facebookresearch/dlrm/blob/main/dlrm_s_pytorch.py
# ******************************************************************************

from typing import List, Optional, Union, Callable

import numpy as np
import torch
from torch import nn

# OverArch topmlp
# DenseArch botmlp
# InteractionDCNArch interaction
# SparseArch Embeddingbags


def _calculate_fan_in_and_fan_out(shape):
    # numpy array version
    dimensions = len(shape)
    assert dimensions >= 2, (
        "Fan in and fan out can not be computed for tensor with fewer than 2 "
        "dimensions"
    )

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if len(shape) > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(shape, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(mode, valid_modes)
        )

    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == "fan_in" else fan_out


def calculate_gain(nonlinearity, param=None):
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (not isinstance(param, bool) and isinstance(param, int)) or isinstance(
            param, float
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return np.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
        # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def xavier_norm_(shape: tuple, gain: float = 1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    mean = 0.0
    d = np.random.normal(mean, std, size=shape).astype(np.float32)
    return d


def kaiming_uniform_(
    shape: tuple, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
    assert 0 not in shape, "Initializing zero-element tensors is a no-op"
    fan = _calculate_correct_fan(shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=shape)


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        bias: bool = True,
        activation: Union[
            str,
            Callable[[], nn.Module],
            nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
        device: Optional[torch.device] = None,
        sigmoid: int = -1,
    ) -> None:
        super().__init__()
        if activation == "relu":
            activation = nn.ReLU
        elif activation == "sigmoid":
            activation = nn.Sigmoid
        layers: nn.ModuleList = nn.ModuleList()
        if not isinstance(activation, str):
            for i in range(len(layer_sizes)):
                m = layer_sizes[i - 1] if i > 0 else in_size
                n = layer_sizes[i]
                LL = nn.Linear(m, n, bias=True)
                W = kaiming_uniform_((n, m), a=np.sqrt(5)).astype(np.float32)
                B = np.zeros(shape=n).astype(np.float32)
                LL.weight.data = torch.tensor(W, requires_grad=True)
                LL.bias.data = torch.tensor(B, requires_grad=True)
                layers.append(LL)
                if i != sigmoid:
                    layers.append(activation())
                else:
                    layers.append(nn.Sigmoid())
        else:
            if activation == "swish_layernorm":
                raise AssertionError("swish layernorm is not supported in mlperf")

        self._mlp: nn.Module = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor of shape (B, I) where I is number of elements
                in each input sample.

        Returns:
            torch.Tensor: tensor of shape (B, O)
            where O is `out_size` of the last Perceptron module.
        """
        return self._mlp(input)


class LowRankCrossNet(nn.Module):
    def __init__(self, in_features: int, num_layers: int, low_rank: int) -> None:
        super().__init__()
        assert low_rank >= 1, "Low rank must be larger or equal to 1"

        self._num_layers = num_layers
        self._low_rank = low_rank
        W_kernels: nn.ParameterList = nn.ParameterList()
        for _i in range(self._num_layers):
            Wnp = xavier_norm_((in_features, self._low_rank))
            Wp = nn.Parameter(torch.tensor(Wnp))
            W_kernels.append(Wp)
        V_kernels: nn.ParameterList = nn.ParameterList()
        for _i in range(self._num_layers):
            Vnp = xavier_norm_((self._low_rank, in_features))
            V_kernels.append(nn.Parameter(torch.tensor(Vnp)))
        bias: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(nn.init.zeros_(torch.empty(in_features)))
                for i in range(self._num_layers)
            ]
        )
        self.MLPs = nn.ModuleDict()
        for i in range(num_layers):
            self.MLPs[f"V{i}"] = nn.Linear(in_features, low_rank, bias=False)
            self.MLPs[f"W{i}"] = nn.Linear(low_rank, in_features, bias=True)
            self.MLPs[f"V{i}"].weight = V_kernels[i]
            self.MLPs[f"W{i}"].weight = W_kernels[i]
            self.MLPs[f"W{i}"].bias = bias[i]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_0 = input
        x_l = x_0
        for layer in range(self._num_layers):
            x_l_v = self.MLPs[f"V{layer}"](x_l)
            x_l_w = self.MLPs[f"W{layer}"](x_l_v)
            x_l = x_0 * x_l_w + x_l  # (B, N)
        return x_l


class MergedEmbeddingBagCat(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings_pool: List[int],
        use_int8: bool = False,
        use_bf16: bool = True,
    ):
        super().__init__()
        self._multi_hot = [
            3,
            2,
            1,
            2,
            6,
            1,
            1,
            1,
            1,
            7,
            3,
            8,
            1,
            6,
            9,
            5,
            1,
            1,
            1,
            12,
            100,
            27,
            10,
            3,
            1,
            1,
        ]
        self._embedding_dim = embedding_dim
        self._num_embeddings = len(num_embeddings_pool)
        embedding_bags: nn.ModuleList = nn.ModuleList()
        for num_embeddings in num_embeddings_pool:
            W = torch.empty(num_embeddings, embedding_dim)
            EE = torch.nn.EmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                _weight=W,
                include_last_offset=True,
                mode="sum",
            )
            embedding_bags.append(EE)
        self.embedding_bags = embedding_bags
        self.weights = tuple([e.weight for e in self.embedding_bags])
        self.int8 = use_int8
        self.bf16 = use_bf16
        return

    def forward(
        self, index: List[torch.Tensor], offset: List[torch.Tensor], dense
    ) -> torch.Tensor:
        self.int8 = False
        B = offset[0].numel() - 1  # it works without -1
        if self.bf16:
            res = [
                e(i.long(), o.long())
                for (e, i, o) in zip(self.embedding_bags, index, offset, strict=False)
            ]
            res = [dense] + res
            data = torch.cat(res, dim=1).reshape(
                B, (self._num_embeddings + 1) * self._embedding_dim
            )
            return data
        else:
            res = []  # removed list comprehension
            for idx in range(len(self.embedding_bags)):
                e, i, o = self.embedding_bags[idx], index[idx], offset[idx]
                res.append(e(i, o))
            res = [dense] + res
            data = torch.cat(res, dim=1).reshape(
                B, (self._num_embeddings + 1) * self._embedding_dim
            )
            return data


class SparseArch(nn.Module):
    """
    Processes the sparse features of DLRM. Does embedding lookups for all EmbeddingBag
    and embedding features of each collection.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): represents a collection of
            pooled embeddings.

    Example::

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(embedding_bag_collection)

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f2"],
           values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
           offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sparse_embeddings = sparse_arch(features)
    """

    def __init__(self, embedding_bag_collection: MergedEmbeddingBagCat) -> None:
        super().__init__()
        self.embedding_bag_collection: MergedEmbeddingBagCat = embedding_bag_collection
        self._sparse_feature_names: List[str] = [
            name for name, param in embedding_bag_collection.named_parameters()
        ]

    def forward(self, index, offset, dense) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """
        return self.embedding_bag_collection(index, offset, dense)

    # return y

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names


class DenseArch(nn.Module):
    """
    Processes the dense features of DLRM model.

    Args:
        in_features (int): dimensionality of the dense input features.
        layer_sizes (List[int]): list of layer sizes.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        dense_arch = DenseArch(10, layer_sizes=[15, D])
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(
            in_features, layer_sizes, bias=True, activation="relu", device=device
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): an input tensor of dense features.

        Returns:
            torch.Tensor: an output tensor of size B X D.
        """
        return self.model(features)


class InteractionDCNArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the output of a Deep Cross Net v2
    https://arxiv.org/pdf/2008.13535.pdf with a low rank approximation for the
    weight matrix. The input and output sizes are the same for this
    interaction layer (F*D + D).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        DCN = LowRankCrossNet(
            in_features = F*D+D,
            dcn_num_layers = 2,
            dnc_low_rank_dim = 4,
        )
        inter_arch = InteractionDCNArch(
            num_sparse_features=len(keys),
            crossnet=DCN,
        )

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (F*D + D)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(
        self, num_sparse_features: int, embedding_dim: int, crossnet: nn.Module
    ) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.D: int = embedding_dim
        self.crossnet = crossnet
        self.ID = (self.F + 1) * self.D

    def forward(self, combined_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (F*D + D).
        """
        return self.crossnet(combined_values)


class OverArch(nn.Module):
    """
    Final Arch of DLRM - simple MLP over OverArch.

    Args:
        in_features (int): size of the input.
        layer_sizes (List[int]): sizes of the layers of the `OverArch`.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        over_arch = OverArch(10, [5, 1])
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError("OverArch must have multiple layers.")
        self.model: nn.Module = MLP(
            in_features,
            layer_sizes,
            bias=True,
            activation="relu",
            device=device,
            sigmoid=4,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor):

        Returns:
            torch.Tensor: size B X layer_sizes[-1]
        """
        return self.model(features)


class DLRMMLPerf(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings_pool: List[int],
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dcn_num_layers: int,
        dcn_low_rank_dim: int,
        use_int8: bool,
        use_bf16: bool,
    ) -> None:
        super().__init__()
        self.sparse_arch: SparseArch = SparseArch(
            MergedEmbeddingBagCat(
                embedding_dim, num_embeddings_pool, use_int8, use_bf16
            )
        )
        self.dense_arch = DenseArch(
            in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes
        )
        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)
        crossnet = LowRankCrossNet(
            in_features=(num_sparse_features + 1) * embedding_dim,
            num_layers=dcn_num_layers,
            low_rank=dcn_low_rank_dim,
        )
        self.inter_arch = InteractionDCNArch(
            num_sparse_features=num_sparse_features,
            embedding_dim=embedding_dim,
            crossnet=crossnet,
        )
        over_in_features: int = (num_sparse_features + 1) * embedding_dim
        self.over_arch = OverArch(
            in_features=over_in_features, layer_sizes=over_arch_layer_sizes
        )
        return

    def forward(self, densex, index, offset) -> torch.Tensor:
        embedded_dense = self.dense_arch(densex)
        combined_values = self.sparse_arch(index, offset, embedded_dense)
        concatenated_dense = self.inter_arch(combined_values)
        out = self.over_arch(concatenated_dense)
        out = torch.reshape(out, (-1,))
        return out
