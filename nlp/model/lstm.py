from typing import Optional, Tuple

import torch
from torch import Tensor

from .base import RNNBase


class LSTMCell(RNNBase):
    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LSTMCell, self).__init__(
            input_size, hidden_size, bias=bias, num_chunks=4, **factory_kwargs
        )

    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (hx, hx)

        hx, cx = hx
        gates = self.ih(input) * self.hh(hx)  # first layer
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        forget_layer = torch.sigmoid(forget_gate)
        input_layer = torch.sigmoid(input_gate)
        adjusted_cell = torch.tanh(cell_gate)
        output_layer = torch.sigmoid(output_gate)

        c_t = cx * forget_layer + input_layer * adjusted_cell
        h_t = output_layer * torch.tanh(c_t)

        return (h_t, c_t)
