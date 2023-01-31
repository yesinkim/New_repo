from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .base import RNNBase, RNNCellBase


class GRUCell(RNNCellBase):
    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(GRUCell, self).__init__(
            input_size, hidden_size, bias=bias, num_chunks=3, **factory_kwargs
        )

    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        if hx is None:
            hx = torch.zeros(
                input.size(0),
                self.num_chunks * self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )

        x_t = self.ih(input)  # -> (3, 60)
        h_t = self.hh(hx)  # -> (3, 60)

        x_reset, x_update, x_new = x_t.chunk(3, 1)  # 각 (3, 20)
        h_reset, h_update, h_new = h_t.chunk(3, 1)  # 각 (3, 20)
        # TODO: 식 다시 확인 (참고링크: https://wikidocs.net/22889)
        r"""
        .. math::
            r_t: reset_gate, z_t: update_gate, n_t = new_gate
            \begin{array}{ll}
                r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
                z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
                n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
                h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
            \end{array}"""
        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_update + h_update)
        new_gate = torch.tanh(reset_gate * (h_new + x_new))

        h_t = (1 - update_gate) * new_gate + update_gate * hx

        return h_t


# TODO: GRU 다른 코드 참고 최~~대한 하지 않고 만들어보기!!!
class GRU(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        # TODO: What's GRU in super.init~???
        super(GRU, self).__init__("GRU", *args, **kwargs)
        self.forward_gru = self.init_layers()
        if self.bidirectional:
            self.backward_gru = self.init_layers()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        batch_dim = 0 if self.batch_first else 1
        sequence_dim = 1 if self.batch_first else 0
        batch_size = input.size(batch_dim)
        seqeunce_length = input.size(sequence_dim)

        is_batch = input.dim() == 3  # input이 3차원 tensor인지?

        if not is_batch:  # batch_dimesion 에러 raise
            input = input.unsqueeze(batch_dim)
            if hx is not None:  # hidden_state를 받으면
                if hx.dim() != 2:  #
                    raise RuntimeError(
                        f"For unbatched 2D input, hx should also be 2D but got {hx.dim()}D tensor"
                    )
                hx = hx.unsqueeze(1)  # tensor.size() -> (num_layers, 1, H_out)
                print(f"hx tensor.size(): {hx.size()}")

        else:
            if (
                hx is not None and hx.dim() != 3
            ):  # hidden_state를 받았는데, dimension이 3이 아닐 때
                raise RuntimeError(
                    f"For batched 3D input, hx should also be 3D but got {hx.dim()}D tensor"
                )

        if hx is None:
            hx = torch.zeros(
                self.num_layers * self.num_direction,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )

        hidden_state = []

        if self.bidirectional:
            next_forward_hidden, next_backward_hidden = [], []
            for layer_idx, (forward_cell, backward_cell) in enumerate(
                zip(self.forward_gru, self.backward_gru)
            ):
                if layer_idx == 0:
                    input_f_state = input
                    input_b_state = input

                else:
                    input_f_state = torch.stack(next_forward_hidden, dim=sequence_dim)
                    input_b_state = torch.stack(next_backward_hidden, dim=sequence_dim)
                    next_forward_hidden, next_backward_hidden = [], []

                forward_cell_i = hx[2 * layer_idx, :, :]
                backward_cell_i = hx[2 * layer_idx + 1, :, :]

                for i in range(seqeunce_length):
                    input_f_i = (
                        input_f_state[:, i, :]
                        if self.batch_first
                        else input_f_state[i, :, :]
                    )
                    input_b_i = (
                        input_b_state[:, -(i + 1), :, :]
                        if self.batch_first
                        else input_b_state[-(i + 1), :, :]
                    )

                    forward_cell_i = forward_cell(input_f_i, forward_cell_i)
                    backward_cell_i = backward_cell(input_b_i, backward_cell_i)

                    if self.dropout:
                        forward_cell_i = self.dropout(forward_cell_i)
                        backward_cell_i = self.dropout(backward_cell_i)

                    next_forward_hidden.append(forward_cell_i)
                    next_backward_hidden.append(backward_cell_i)

                hidden_state.append(torch.stack(next_forward_hidden, dim=sequence_dim))
                hidden_state.append(
                    torch.stack(next_backward_hidden[::-1], dim=sequence_dim)
                )

            hidden_states = torch.stack(hidden_state, dim=0)
            output_f_state = hidden_states[-2, :, :, :]
            output_b_state = hidden_states[-1, :, :, :]
            output = torch.cat([output_f_state, output_b_state], dim=2)

        else:  # non_bidirectional
            next_hidden = []
            for layer_idx, gru_cell in enumerate(self.forward_gru):
                if layer_idx == 0:
                    input_state = input
                else:
                    input_state = torch.stack(next_hidden, dim=sequence_dim)
                    next_hidden = []

                h_i = hx[layer_idx, :, :]

                for i in range(seqeunce_length):
                    x_i = (
                        input_state[:, i:, :]
                        if self.batch_first
                        else input_state[i, :, :]
                    )
                    print(f"i_{i+1}: {x_i.size()}")
                    h_i = gru_cell(
                        x_i, h_i
                    )  # x_i.size -> (5, 3, 10), h_i.size -> 3, 20
                    print(f"h_{i+1}: {h_i.size()}")
                    if self.dropout:
                        h_i = self.dropout(h_i)
                    next_hidden.append(h_i)
                hidden_state.append(torch.stack(next_hidden, dim=sequence_dim))
            hidden_states = torch.stack(hidden_state, dim=0)
            output = hidden_states[-1, :, :, :]  # the lastest layer
        hn = (
            hidden_states[:, :, -1, :]
            if self.batch_first
            else hidden_states[:, -1, :, :]
        )

        return output, hn

    def init_layers(self) -> List[GRUCell]:
        """상속받은 클래스의 init_layers 수행"""
        layers = []
        for layer_idx in range(self.num_layers):
            input_size = self.input_size if layer_idx == 0 else self.hidden_size
            layers.append(
                GRUCell(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        return layers
