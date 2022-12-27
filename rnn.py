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
        self.num_chunks = num_chunks
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
        device = None,
        dtype = None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size=input_size, hidden_size=hidden_size, bias=bias, num_chunks=1,  **factory_kwargs)
        

        def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
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


# nn.LSTMCell 
class LSTMCell(RNNBase):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        bias: bool,
        device = None,
        dtype = None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LSTMCell, self).__init__(input_size, 4 * hidden_size, bias=bias, num_chunks=4, **factory_kwargs)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size(0), self.num_chunks * self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (hx, hx)
        
        hx, cx = hx
        gates = self.ih(input) * self.hh(hx)    # first layer
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        c_t = cx * f_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return (h_t, c_t)


if __name__ == '__main__':
    lstm = LSTMCell(10, 20, True)
    input = torch.randn(2, 3, 10)
    lstm(input)