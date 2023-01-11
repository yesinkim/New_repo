from typing import List, Optional

import torch
from torch import Tensor

from .base import RNNBase, RNNCellBase

torch.nn.RNN


class RNNCell(RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        """An Elman RNN cell with tanh or ReLU non-linearity.

        Args:
            input_size: The number of expected features in the input 'x'
            hidden_size: The number of fetures in the hidden state 'h'
            bias: If `False` then the layer doen not use bias weight bias.
            nonlinearity: The non-linearity function to use. Can be either `tanh` or `relu`. Defaults to 'tanh'.

        Inputs: input, hidden
            - input: tensor containing the input features
            - hidden: tensor containing the initial hidden state
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            num_chunks=1,
            **factory_kwargs,
        )

        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Non-linearity must be one of 'tanh', 'relu'.")

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        # Make zero tensor if tensor is not initialized
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )

        # forward
        hy = self.ih(input) * self.hh(hx)

        # function
        if self.nonlinearity == "tanh":
            ret = torch.tanh(hy)

        else:
            ret = torch.relu(hy)

        return ret


# TODO: dimension 변화 확인해보고 체크하기
class RNN(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        if self.nonlinearity == "tanh":
            mode = "RNN_TANH"

        elif self.nonlinearity == "relu":
            mode = "RNN_RELU"

        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
        super(RNN, self).__init__(mode, *args, **kwargs)

        self.forward_rnn = self.init_layers()
        if self.bidirectional:
            self.backward_rnn = self.init_layers()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """_summary_

        N: batch_size
        L: sequence_length
        D: 2 if bidirectional=True, otherwise 1
        H_in: input_size
        H_out: hidden_size

        Args:
            input (Tensor): The input can also be a packed variable length sequence.
                - (L, H_in) for unbatched input
                - (N, L, H_in) for batched input when batch first == True (모델한테 batch_size위치를 알려줌)
                - (L, N, H_in) for batched input when batch first == False
            hx (Optional[Tensor], optional):
                - (num_layers, H_out)
                - (num_layers, batch_size, H_out)

        Returns:
            Tensor: The output of the RNN.
        """

        batch_dim = 0 if self.batch_first else 1
        sequence_dim = 1 if self.batch_first else 0
        is_batch = input.dim() == 3  # batch가 포함된 3차원 텐서
        if not is_batch:
            input = input.unsqueeze(batch_dim)
            if hx is not None:  # hidden_state를 받으면
                if hx.dim() != 2:  #
                    raise RuntimeError(
                        f"For unbatched 2D input, hx should also be 2D but got {hx.dim()}D tensor"
                    )
                hx = hx.unsqueeze(1)  # tensor.size() -> (num_layers, 1, H_out)

        else:
            if (
                hx is not None and hx.dim() != 3
            ):  # hidden_state를 받았는데, dimension이 3이 아닐 때
                raise RuntimeError(
                    f"For batched 3D input, hx should also be 3D but got {hx.dim()}D tensor"
                )

        # input.size() -> (batch_size, sequence_length, input_size)
        # hx.size() -> (num_layers, batch_size, hidden_size) or None (becuase hx is Optional)

        batch_size = input.size(0) if self.batch_first else input.size(1)
        sequence_length = input.size(1) if self.batch_first else input.size(0)

        if hx is None:
            hx = torch.zeros(
                self.num_layers * self.num_direction,
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )

        # hx.size() 는 항상 -> (num_layers, batch_size, hidden_size)

        hidden_state = []
        # TODO: 차원이 바뀌는 지점에서 차원 기입
        # 1. batch_first = False, Bidirectional = True
        # 2. batch_first = True, bidirectional = True
        if self.bidirectional:
            next_hidden_forward, next_hidden_backward = [], []
            for layer_idx, (forward_cell, backward_cell) in enumerate(
                self.forward_rnn, self.backward_rnn
            ):
                if layer_idx == 0:
                    input_f_state = input
                    input_b_state = input
                else:
                    input_f_state = torch.stack(next_hidden_forward, dim=sequence_dim)
                    input_b_state = torch.stack(next_hidden_backward, dim=sequence_dim)
                    next_hidden_forward, next_hidden_backward = [], []

                forward_cell_i = hx[2 * layer_idx, :, :]
                backward_cell_i = hx[2 * layer_idx + 1, :, :]

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
                    )  # 완전히 뒤집어지는 backward이심.

                    h_f_i = forward_cell(input_f_i, h_f_i)
                    h_b_i = backward_cell(input_b_i, h_b_i)

                    if self.dropout:
                        h_f_i = self.dropout(h_f_i)
                        h_b_i = self.dropout(h_b_i)

                    next_hidden_forward.append(h_f_i)
                    next_hidden_backward.append(h_b_i)

                hidden_state.append(next_hidden_forward, dim=sequence_dim)
                hidden_state.append(
                    next_hidden_backward[::-1], dim=sequence_dim
                )  # backward이기 때문에 뒤집어진다~

            hidden_states = torch.stack(
                hidden_state, dim=0
            )  # -> (num_layers * 2, batch_size, sequence_length, hidden_out)

            output_f_state = hidden_states[-2, :, :, :]
            output_b_state = hidden_states[-1, :, :, :]
            output = torch.cat([output_f_state, output_b_state])

        else:
            next_hidden = []
            for layer_idx, rnn_cell in enumerate(self.forward_rnn):
                if layer_idx == 0:
                    input_state = input
                else:
                    input_state = torch.stack(
                        next_hidden, dim=sequence_dim
                    )  # -> (sequence_length, batch_size, hidden_size)
                    next_hidden = []
                input_state = (
                    input  # TODO: 134번째 줄에서 에러 발생. 조건 필요 (layer 갯수가 2일 때 어떻게 될까요~~?)
                )
                h_i = hx[
                    layer_idx, :, :
                ]  # [layer_idx, batch_size, H_out]: idx-1번째의 hidden_state

                for i in range(sequence_length):
                    x_i = (
                        input_state[:, i, :]
                        if self.batch_first
                        else input_state[i, :, :]
                    )  # 각 sequence tensor들을 골라냄 batch_first일 때 []
                    h_i = rnn_cell(x_i, h_i)
                    if self.dropout:
                        h_i = self.dropout(h_i)
                    next_hidden.append(h_i)  # 각 층의 hidden_states
                hidden_state.append(
                    torch.stack(
                        next_hidden, dim=sequence_dim
                    )  # -> (sequence_length, batch_size, hidden_out)
                )
                # size() -> [batch_size, hidden_size * sequence_length, hidden_size * sequence_length
            hidden_states = torch.stack(
                hidden_state, dim=0
            )  # -> (num_layers, sequence_length, batch_size, hidden_out)
            output = hidden_states[
                -1, :, :, :
            ]  # -> (sequence_length, batch_size, hidden_out)
            if self.batch_first:
                hn = hidden_state[
                    :, :, -1, :
                ]  # -> (num_layers, batch_size, hidden_out)
            else:
                hn = hidden_state[
                    :, -1, :, :
                ]  # -> (num_layers, batch_size, hidden_out)

            return output, hn

    def init_layers(self) -> List[RNNCellBase]:
        """상속받은 클래스의 init_layers 수행"""
        layers = []
        for layer_idx in range(self.num_layers):
            input_size = self.input_size if layer_idx == 0 else self.hidden_size
            layers.append(
                RNNCell(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    nonlinearity=self.nonlinearity,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        return layers
