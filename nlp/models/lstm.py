from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import RNNBase, RNNCellBase


class LSTMCell(RNNCellBase):
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


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        super(LSTM, self).__init__("LSTM", *args, **kwargs)
        self.forward_lstm = self.init_layers()
        if self.bidirectional:
            self.backward_lstm = self.init_layers()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        batch_dim = 0 if self.batch_first else 1
        sequence_dim = 1 if self.batch_first else 0
        batch_size = input.size(0) if self.batch_first else input.size(1)
        sequence_length = input.size(1) if self.batch_first else input.size(0)
        print(f"batch_size: {batch_size}")
        print(f"sequence_size: {sequence_length}")
        is_batch = input.dim() == 3  # batch가 포함된 3차원 텐서

        if not is_batch:
            input = input.unsqueeze(batch_dim)

        if hx is None:
            h_zero = torch.zeros(
                self.num_layers * self.num_direction,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            c_zero = torch.zeros(
                self.num_layers * self.num_direction,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            hx = (h_zero, c_zero)

        elif is_batch:  # batch가 존재할 때
            if hx[0].dim() != 3 or hx[1].dim() != 3:
                msg = f"For unbatched 3D input, hx should also be 3D but got {hx.dim()}D tensor"
                raise RuntimeError(msg)

        else:
            if hx[0].dim() != 2 or hx[1].dim() != 2:
                msg = f"For unbatched 2D input, hx should also be 2D but got {hx.dim()}D tensor"
                raise RuntimeError(msg)

            hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

        hidden_state = []
        cell_state = []

        if self.bidirectional:
            next_hidden_forward, next_hidden_backward = [], []
            next_cell_forward, next_cell_backward = [], []
            for layer_idx, (forward_cell, backward_cell) in enumerate(
                zip(self.forward_lstm, self.backward_lstm)
            ):
                if layer_idx == 0:
                    input_f_state = input
                    input_b_state = input

                else:
                    input_f_state = torch.stack(next_hidden_forward, dim=sequence_dim)
                    input_b_state = torch.stack(next_hidden_backward, dim=sequence_dim)
                    next_hidden_forward, next_hidden_backward = [], []
                    next_cell_forward, next_cell_backward = [], []

                forward_h_i = hx[0][2 * layer_idx, :, :]
                backward_h_i = hx[0][2 * layer_idx + 1, :, :]
                forward_c_i = hx[1][2 * layer_idx, :, :]
                backward_c_i = hx[1][2 * layer_idx + 1, :, :]

                for i in range(sequence_length):
                    input_f_i = (
                        input_f_state[:, i, :]
                        if self.batch_first
                        else input_f_state[i, :, :]
                    )
                    input_b_i = (
                        input_b_state[:, -(i + 1), :]
                        if self.batch_first
                        else input_b_state[-(i + 1), :, :]
                    )

                    forward_h_i, forward_c_i = forward_cell(
                        input_f_i, (forward_h_i, forward_h_i)
                    )
                    backward_h_i, backward_c_i = backward_cell(
                        input_b_i, (forward_c_i, forward_c_i)
                    )

                    if self.dropout:
                        forward_h_i = self.dropout(forward_h_i)
                        backward_h_i = self.dropout(backward_h_i)
                        forward_c_i = self.dropout(forward_c_i)
                        backward_c_i = self.dropout(backward_c_i)

                    next_hidden_forward.append(forward_h_i)
                    next_hidden_backward.append(backward_h_i)
                    next_cell_forward.append(forward_c_i)
                    next_cell_backward.append(backward_c_i)

                hidden_state.append(torch.stack(next_hidden_forward, dim=sequence_dim))
                hidden_state.append(torch.stack(next_cell_backward[::-1], dim=sequence_dim))
                cell_state.append(torch.stack(next_cell_forward, dim=sequence_dim))
                cell_state.append(torch.stack(next_cell_backward[::-1], dim=sequence_dim))
            
            hidden_states = torch.stack(hidden_state, dim=0)
            cell_states =  torch.stack(cell_state, dim=0)

            output_f_state = hidden_states[-2, :, :, :]
            output_b_state = hidden_states[-1, :, :, :]
            output = torch.cat([output_f_state, output_b_state], dim=2)

        else:
            next_hidden, next_cell = [], []
            for layer_idx, lstm_cell in enumerate(self.forward_lstm):
                if layer_idx == 0:
                    input_state = input
                else:
                    input_state = torch.stack(next_hidden, dim=sequence_dim)
                    next_hidden = []
                    next_cell = []
                # lstm에는 rnn에서 cell_state가 추가 되었음 (c를 중심으로 확인 해볼 것.)
                h_i = hx[0][layer_idx, :, :]
                c_i = hx[1][layer_idx, :, :]

                for i in range(sequence_length):
                    input_i = (
                        input_state[:, i, :]
                        if self.batch_first
                        else input_state[i, :, :]
                    )
                    h_i, c_i = lstm_cell(input_i, (h_i, c_i))
                    if self.dropout:
                        h_i = self.dropout(h_i)
                        c_i = self.dropout(c_i)

                    next_hidden.append(h_i)
                    next_cell.append(c_i)

                hidden_state.append(torch.stack(next_hidden, dim=sequence_dim))
                cell_state.append(torch.stack(next_cell, dim=sequence_dim))

            hidden_states = torch.stack(hidden_state, dim=0)
            cell_states = torch.stack(cell_state, dim=0)

            output = hidden_states[-1, :, :, :]

        h_n = (
            hidden_states[:, :, -1:, :]
            if self.batch_first
            else hidden_states[:, -1, :, :]
        )
        c_n = (
            cell_states[:, :, -1:, :]
            if self.batch_first
            else cell_states[:, -1, :, :]
        )
        return output, (h_n, c_n)

    def init_layers(self) -> List[LSTMCell]:
        """상속받은 클래스의 init_layers 수행"""
        layers = []
        for layer_idx in range(self.num_layers):
            input_size = self.input_size if layer_idx == 0 else self.hidden_size
            layers.append(
                LSTMCell(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        return layers
