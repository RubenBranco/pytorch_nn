from math import sqrt
import random
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if bidirectional:
            raise NotImplementedError("Bidirectional RNN not currently implemented")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device
        self.dtype = dtype

        self.weight_ih_l = nn.ParameterList()
        self.weight_hh_l = nn.ParameterList()

        if bias:
            self.bias_ih_l = nn.ParameterList()
            self.bias_hh_l = nn.ParameterList()
        else:
            self.register_parameter("bias_ih_l", None)
            self.register_parameter("bias_hh_l", None)

        # All the weights and biases are initialized from uniform distribution
        # U(-sqrt(k), sqrt(k)), where k = 1 / hidden_size

        k = 1 / sqrt(self.hidden_size)

        for i in range(self.num_layers):
            if i == 0:
                ih_size = (hidden_size, input_size)
            else:
                ih_size = (hidden_size, hidden_size)

            self.weight_ih_l.append(
                nn.Parameter(torch.empty(ih_size, device=device, dtype=dtype))
            )
            self.weight_hh_l.append(
                nn.Parameter(
                    torch.empty((hidden_size, hidden_size), device=device, dtype=dtype)
                )
            )
            with torch.no_grad():
                self.weight_ih_l[i].uniform_(-k, k)
                self.weight_hh_l[i].uniform_(-k, k)

            if bias:
                self.bias_ih_l.append(
                    nn.Parameter(torch.empty((hidden_size), device=device, dtype=dtype))
                )
                self.bias_hh_l.append(
                    nn.Parameter(torch.empty((hidden_size), device=device, dtype=dtype))
                )
                with torch.no_grad():
                    self.bias_ih_l[i].uniform_(-k, k)
                    self.bias_hh_l[i].uniform_(-k, k)

    def forward(self, x: Tensor, h: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if not self.batch_first:
            x = x.transpose(1, 0)

        if h is None:
            h_previous = torch.zeros(
                (self.num_layers, x.size(0), self.hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            h_previous = h

        h_t = h_previous
        h_history = torch.zeros(
            (x.size(0), x.size(1), self.hidden_size),
            device=self.device,
            dtype=self.dtype,
        )

        for t in range(x.size(1)):
            for l in range(self.num_layers):
                h_t_layer = einsum(
                    x[:, t] if l == 0 else h_t[l - 1],
                    self.weight_ih_l[l],
                    "b d, c d -> b c",
                ) + einsum(h_previous[l], self.weight_hh_l[l], "b d, c d -> b c")

                if self.bias:
                    h_t_layer += self.bias_ih_l[l] + self.bias_hh_l[l]

                if self.nonlinearity == "tanh":
                    h_t_layer = torch.tanh(h_t_layer)
                elif self.nonlinearity == "relu":
                    h_t_layer = torch.relu(h_t_layer)

                h_t[l] = h_t_layer

            h_history[:, t] = h_t[-1]
            h_previous = h_t

        if not self.batch_first:
            h_history = h_history.transpose(1, 0)

        return h_history, h_t
