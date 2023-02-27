import random
from ast import Tuple
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        mode: str = "lstm",
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.layers = self.select_mode(
            mode=mode,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bidirectional=bidirectional,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
        )

        for m in self.modules():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

    def forward(
        self,
        enc_input: Tensor,
        enc_hidden: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ):
        embeded = self.embedding(enc_input)
        output, hidden = self.layers(embeded, enc_hidden)
        return output, hidden

    def select_mode(
        self,
        mode: str,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ):
        if mode == "lstm":
            return nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        elif mode == "rnn":
            return nn.RNN(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        elif mode == "gru":
            return nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        else:
            raise ValueError("param `mode` must be one of [rnn, lstm, gru]")


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        dropout: float,
        mode: str = "lstm",
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.layers = self.select_mode(
            mode, hidden_size, n_layers, dropout, batch_first, bias, bidirectional
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, dec_input: Tensor, hidden=None):
        embeded = self.embedding(dec_input)
        relu_embeded = F.relu(embeded)  # 이렇게 했더니 더 좋았다!
        output, hidden = self.layers(relu_embeded, hidden)
        output = self.softmax(self.linear(output))
        return output, hidden

    def select_mode(
        self,
        mode: str,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        batch_first: bool = True,
        bias: bool = True,
        bidirectional: bool = False,
    ):

        if mode == "lstm":
            return nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        elif mode == "gru":
            return nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )

        elif mode == "rnn":
            return nn.RNN(
                hidden_size,
                hidden_size,
                num_layers=n_layers,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                batch_first=batch_first,
            )


class Seq2Seq(nn.Module):
    def __init__(
        self,
        enc_inputsize: int,
        dec_inputsize: int,
        d_hidden: int,
        n_layers: int,
        max_seq_length: int,
        mode: str = "lstm",
        dropout_rate: float = 0.0,
        bidirectional: bool = True,
        bias: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_size=enc_inputsize,
            hidden_size=d_hidden,
            n_layers=n_layers,
            dropout=dropout_rate,
            mode=mode,
            bidirectional=bidirectional,
            bias=bias,
            batch_first=batch_first,
        )
        self.decoder = Decoder(
            output_size=dec_inputsize,
            hidden_size=d_hidden,
            n_layers=n_layers,
            dropout=dropout_rate,
            mode=mode,
            bidirectional=bidirectional,
            bias=bias,
            batch_first=batch_first,
        )
        self.max_seq_length = max_seq_length
        self.vocab_size = dec_inputsize

    def forward(
        self,
        enc_input: Tensor,
        dec_input: Tensor,
        teacher_forcing_rate: Optional[float] = 1.0,
    ) -> Tensor:
        enc_hidden = None
        for i in range(self.max_seq_length):
            enc_input_i = enc_input[:, i]
            _, enc_hidden = self.encoder(enc_input_i, enc_hidden)

        decoder_output = torch.zeros(
            dec_input.size(0), self.max_seq_length, self.vocab_size
        )
        dec_hidden = enc_hidden
        for i in range(self.max_seq_length):
            if i == 0 or random.random() >= teacher_forcing_rate:
                dec_input_i = dec_input[:, i]  # 정답을 넣어줌
            else:
                dec_input_i = dec_output_i.topk(1)[1].sequeeze().detach()
            dec_output_i, dec_hidden = self.decoder(dec_input_i, dec_hidden)
            decoder_output[:, i, :] = dec_output_i
        return decoder_output
