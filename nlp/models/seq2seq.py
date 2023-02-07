import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

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
            bidirectional: bool = False) -> None:
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.layers = self.select_mode(mode, hidden_size, n_layers, dropout, batch_first, bias, bidirectional)

    def forward(self, enc_input: Tensor):
        embeded = self.embedding(enc_input)
        output, hidden = self.layers(embeded)
        return output, hidden

    def select_mode(
            self,
            mode: str,
            hidden_size: int,
            n_layers: int,
            dropout: float,
            batch_first: bool = True,
            bias: bool = True,
            bidirectional: bool = False):

        if mode == 'lstm':
            return nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)
        
        elif mode == 'gru':
            return nn.GRU(hidden_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)
        
        elif mode == 'rnn':
            return nn.RNN(hidden_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)


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
            bidirectional: bool = False) -> None:
        self.embedding = nn.Embedding(hidden_size, output_size)
        self.layers = self.select_mode(mode, hidden_size, n_layers, dropout, batch_first, bias, bidirectional)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_input: Tensor, hi):
        embeded = self.embedding(enc_input)
        relu_embeded = F.relu(embeded)  # 이렇게 했더니 더 좋았다!
        output, hidden = self.layers(relu_embeded)
        output = self.softmax(self.linear(output[0]))
        return output, hidden

    def select_mode(
            self,
            mode: str,
            hidden_size: int,
            n_layers: int,
            dropout: float,
            batch_first: bool = True,
            bias: bool = True,
            bidirectional: bool = False):

        if mode == 'lstm':
            return nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)
        
        elif mode == 'gru':
            return nn.GRU(hidden_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)
        
        elif mode == 'rnn':
            return nn.RNN(hidden_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional, bias=bias, dropout=dropout, batch_first=batch_first)
