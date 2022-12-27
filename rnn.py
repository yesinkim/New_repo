from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class RNNBase(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        bias: bool,
        num_chunks: int,
        nonlinearity: str ='tanh',
        device = None,
        dtype = None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ['tanh', 'relu']:
            raise ValueError("Invalid nonlinearity selected for RNN. Can be either  ``tanh`` or ``relu``")

        self.ih = nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias, **factory_kwargs)
        self.hh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias, **factory_kwargs)



class RNNCell(RNNBase):
    # nn.RNN
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        bias: bool,
        nonlinearity: str ='tanh',
        device = None,
        dtype = None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size=input_size, hidden_size=hidden_size, bias=bias, **factory_kwargs)
        

        def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor]:
            # make zero tensor
            if hx is None:
                hx = torch.zeros(input.size(0), self.hidden_size, dtype=self.dtype)
            
            # forward
            hy = self.ih(input) * self.hh(hx)

            # function
            if self.nonlinearity == 'tanh':
                ret = torch.tanh(hy)
            
            else:
                ret = torch.relu(hy)

            return ret



# class LSTMCell(RNNCell)

if __name__ == '__main__':
    RNNCell(5, 12, bias)