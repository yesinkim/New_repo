import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_positional_encoding_table(max_sequence_size: int, d_hidden: int):
    def get_angle(position: int, i: int) -> float:
        return position / torch.pow(10000, 2 * (i // 2) / d_hidden)

    def get_angle_vector(position: int) -> list[float]:
        return [get_angle(position, hid_j) for hid_j in range(d_hidden)]

    pe_table = torch.Tensor(
        [get_angle_vector(pos_i for pos_i in range(max_sequence_size))]
    )
    pe_table[:, 0::2] = torch.sin(pe_table[:, 0::2])  # 0에서 2씩(짝수만) sin함수에 적용해줌
    pe_table[:, 1::2] = torch.cos(pe_table[:, 1::2])

    return pe_table


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim: int, dropout_rate: float = 0.0) -> None:
        """_summary_

        Args:
            head_dim (int): 각 head에서 사용할 tensor의 차원
            dropout_rate (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = head_dim**0.5

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        score = torch.matmul(query.unsqueeze(1), key.transpose(-1, -2)) / self.scale
        attn_prob = self.dropout(nn.Softmax(dim=-1)(score))
        context = torch.matmul(attn_prob, value).squeeze()

        return context, attn_prob


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_hidden: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        max_sequence_size: int,
        padding_id: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.src_emb = nn.Embedding(input_dim, d_hidden)
        pe_table = get_positional_encoding_table(max_sequence_size, d_hidden)
        self.pe_emb = nn.Embedding.from_pretrained(pe_table, freeze=True)

    def forward(self, enc_input: Tensor):
        position = self.get_position(enc_input)
        combinbed_emb = self.src_emb + self.pe_emb(position)

    def get_position(self, enc_input: Tensor) -> Tensor:
        """_summary_

        Args:
            enc_input (Tensor): size: (batch_size, max_seq_len)

        Returns:
            Tensor: _description_
        """
        position = (
            torch.arange(
                enc_input.size(1), device=enc_input.device, dtype=enc_input.dtype
            )
            .expand(enc_input.size(0), enc_input.size(1))
            .contiguous()
            + 1  # size: (batch_size, max_seq_len)
        )
        pos_mask = enc_input.eq(
            self.padding_id
        )  # if enc_input == padding: True, not: False
        position.masked_fill_(pos_mask, 0)

        return position


class Transformer(nn.Module):
    def __init__(
        self,
        enc_d_input: int,  # source language vocab size
        enc_layers: int,
        enc_heads: int,
        enc_head_dim: int,
        enc_ff_dim: int,
        dec_d_input: int,  # target language vocab size
        dec_layers: int,
        dec_heads: int,
        dec_head_dim: int,
        dec_ff_dim: int,
        d_hidden: int,
        max_sequence_size: int,
        padding_id: int = 3,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_dim=enc_d_input,
            d_hidden=d_hidden,
            n_layers=enc_layers,
            n_heads=enc_heads,
            ff_dim=enc_ff_dim,
            max_sequence_size=max_sequence_size,
            padding_id=padding_id,
            dropout_rate=dropout_rate,
        )


def positional_encoding(max_sequence, hidden_dim) -> Tensor:
    """
    $PE_{(pos,2i)}=\sin \left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)$
    $PE_{(pos,2i+1)}=\cos \left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)$

    Args:
        max_sequence (_type_): _description_
        hidden_dim (_type_): _description_

    Returns:
        Tensor: _description_
    """

    # 숫자 오버 플로우를 방지하기 위해서 log space에서 연산
    # 앞 뒤 연산 시에 dimension, tensor nograds
    pe = torch.zeros(max_sequence, hidden_dim)
    position = torch.arange(0, max_sequence)

    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2)
        * -(torch.log(torch.Tensor([10000])))
        / hidden_dim
    )  # 이거는 그냥 transformer에서 sprt(d_momel)로 나눠주기 때문에 해주는 듯함. (수식에 있음)

    pe[:, 0::2] = torch.sin(position * div_term)  # 0에서 2씩(짝수만) sin함수에 적용해줌
    pe[:, 1::2] = torch.cos(position * div_term)  # 1에서 2씩 증가하며(홀수만) cos함수에 적용

    return pe


class AddNorm(nn.Module):
    """layer 연결시 Residual Connection과 Layer Normalization 수행"""

    def __init__(self, d_model, dropout: float = 0.0):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
