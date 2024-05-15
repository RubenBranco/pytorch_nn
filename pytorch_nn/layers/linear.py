from math import sqrt

import torch
import torch.nn as nn
from einops import einsum
from torch import Tensor


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self._init_params()

    def _init_params(self):
        # weight
        # The values are initialized from U(-sqrt(k), sqrt(k)) where k = 1 / in_features
        k = 1 / self.in_features
        with torch.no_grad():
            self.weight.uniform_(-sqrt(k), sqrt(k))

        # bias
        # The values are initialized from U(-sqrt(k), sqrt(k)) where k = 1 / in_features

        if self.bias is not None:
            with torch.no_grad():
                self.bias.uniform_(-sqrt(k), sqrt(k))

    def forward(self, x: Tensor) -> Tensor:
        x = einsum(x, self.weight, "b d, c d -> b c")

        if self.bias is not None:
            x += self.bias

        return x


class Bilinear(nn.Module):
    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in1_features, in2_features), device=device, dtype=dtype
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self._init_params()

    def _init_params(self):
        # weight
        # The values are initialized from U(-sqrt(k), sqrt(k)) where k = 1 / in1_features
        k = 1 / self.in1_features
        with torch.no_grad():
            self.weight.uniform_(-sqrt(k), sqrt(k))

        # bias
        # The values are initialized from U(-sqrt(k), sqrt(k)) where k = 1 / in1_features
        if self.bias is not None:
            with torch.no_grad():
                self.bias.uniform_(-sqrt(k), sqrt(k))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = einsum(x1, x2, self.weight, "b d, b k, c d k -> b c")

        if self.bias is not None:
            x += self.bias

        return x
