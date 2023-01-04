import math
import numbers
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# nn.RNN
class RNNCellBase(nn.Module):
    """Base class for advanced RNN Module"""

    __constants__ = ["input_size", "output_size", "bias"]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError(
                "Invalid nonlinearity selected for RNN. Can be either  ``tanh`` or ``relu``"
            )

        self.ih = nn.Linear(
            in_features=input_size,
            out_features=num_chunks * hidden_size,
            bias=bias,
            **factory_kwargs
        )

        self.hh = nn.Linear(
            in_features=hidden_size,
            out_features=num_chunks * hidden_size,
            bias=bias,
            **factory_kwargs
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.input_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


class RNNBase(nn.Module):
    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) != 0 else None
        self.bidirectional = bidirectional
        self.device = device
        self.dtype = dtype
        self.num_direction = 2 if bidirectional else 1

        if (
            not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1
        ):  # number가 아닌 값이 들어왔을 때
            raise ValueError("dropout must be a number in the range [0, 1]")

        if dropout >= 0 and num_layers == 1:
            warnings.warn(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                "num_layers greater than 1, but got dropout={} and "
                "num_layers={}".format(dropout, num_layers)
            )

    def init_layers(self):
        pass
