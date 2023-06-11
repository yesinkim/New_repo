# reference: https://paul-hyun.github.io/transformer-01/
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.utils import get_device


def get_positional_encoding_table(max_seq_len: int, hidden_dim: int):
    """_summary_

    Args:
        max_seq_len (int): _description_
        hidden_dim (int): _description_
    """

    def get_angle(position: int, i: int) -> float:
        return position / np.power(10000, 2 * (i // 2) / hidden_dim)

    def get_angle_vector(position: int) -> list[float]:
        return [get_angle(position, hid_j) for hid_j in range(hidden_dim)]

    pe_table = torch.Tensor([get_angle_vector(pos_i) for pos_i in range(max_seq_len)])
    pe_table[:, 0::2] = torch.sin(pe_table[:, 0::2])  # 0에서 2씩(짝수만) sin함수에 적용해줌
    pe_table[:, 1::2] = torch.cos(pe_table[:, 1::2])

    return pe_table


def get_position(sequence: Tensor) -> Tensor:
    """embedding vector의 위치 index를 반환

    Args:
        enc_input (Tensor): size: (batch_size, max_seq_len)

    Returns:
        Tensor: batch_size,
    """
    position = (
        torch.arange(sequence.size(1), device=sequence.device, dtype=sequence.dtype)
        .expand(sequence.size(0), sequence.size(1))
        .contiguous()  # size: (batch_size, max_seq_len)
    )

    return position  # (batch_size, max_seq_len)


def get_padding_mask(seq: Tensor, padding_id: int) -> Tensor:
    """attention mask for pad token

    To avoid using `pad token` in operations,
    return the masking table tensor to use when using masked_fill

    Args:
        seq_q (Tensor): input query tensor
                        tensor.size-> (batch_size, max_seq_len)
        seq_k (Tensor): input key tensor
                        tensor.size-> (batch_size, max_seq_len)
        padding_id (int)

    Returns:
        Tensor: attention pad masking table
    """
    pad_attn_mask = seq.data.eq(padding_id).unsqueeze(
        1
    )  # padding id와 같은 mask에 1번째 자리 차원 증가 size(batch_size, 1, max_seq_len)

    return pad_attn_mask.expand(seq.size(0), seq.size(1), seq.size(1)).contiguous()


def get_look_ahead_mask(seq: Tensor) -> Tensor:
    """t시점의 Tensor가 t+1 tensor를 참조하지 못하도록 ahead token을 masking

    Args:
        seq (Tensor): _description_

    Returns:
        Tensor: _description_
    """
    look_ahead_mask = (
        torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    )  # (batch_size, max_seq_len, max_seq_len) size의 텐서를 생성한 뒤,
    look_ahead_mask = look_ahead_mask.triu(diagonal=1)  # triu로 대각선 기준으로 상삼각형을 살려둔다.

    return look_ahead_mask.eq(1)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim: int, dropout_rate: float = 0.0) -> None:
        """Scaled Dot Product Attention

        Args:
            head_dim (int): 각 head에서 사용할 tensor의 차원
            dropout_rate (float, optional): Literally dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = head_dim**0.5

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        score = (
            torch.matmul(query, key.transpose(-1, -2)) / self.scale
        )  # size: (batch_size, max_seq_len(len_query), max_seq_len(len_key))
        score.masked_fill_(attn_mask, -1e9)
        attn_prob = self.dropout(nn.Softmax(dim=-1)(score))
        context = torch.matmul(attn_prob, value).squeeze()

        return context, attn_prob


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, head_dim, dropout=0) -> None:
        super().__init__()
        self.W_Q = nn.Linear(hidden_dim, n_heads * head_dim)
        self.W_K = nn.Linear(hidden_dim, n_heads * head_dim)
        self.W_V = nn.Linear(hidden_dim, n_heads * head_dim)

        self.self_attn = ScaledDotProductAttention(head_dim, dropout)
        self.linear = nn.Linear(n_heads * head_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor):
        """MultiHeadAttention

        Args:
            query (Tensor): input word vector (size: batchsize, max_seq_len(q), hidden_dim)
            key (Tensor): input word vector (size: batchsize, max_seq_len(k), hidden_dim)
            value (Tensor): input word vector (size: batchsize, max_seq_len(v), hidden_dim)
            attn_mask (Tensor): attention mask (size: batchsize, max_seq_len(q=k=v), hidden_dim)

        Returns:
            _type_: _description_
        """
        batch_size = query.size(0)

        seq_q = (
            self.W_Q(query)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, n_heads, len_q, head_dim)
        seq_s = (
            self.W_K(key)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, n_heads, len_k, head_dim)
        seq_v = (
            self.W_V(value)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, n_heads, len_v, head_dim)
        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.n_heads, 1, 1
        )  # (batch_size, n_heads, len_q, len_k)
        context, _ = self.self_attn(seq_q, seq_s, seq_v, attn_mask)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.head_dim)
        )  # (batch_size, len_1, n_heads * head_dim)

        output = self.linear(context)  # (batch_size, len_q, hidden_dim)
        output = self.dropout(output)

        return output


class PositionWiseFeedForwardNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.func1 = nn.Linear(hidden_dim, ff_dim)
        self.func2 = nn.Linear(ff_dim, hidden_dim)
        self.active = F.relu
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs => (batch_size, feedforward_dim(d_ff), max_seq_len)
        output = self.dropout(
            self.active(self.func1(inputs))
        )  # (batch_size, max_seq_len, hidden_dim)
        output = self.func2(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout_rate: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            hidden_dim=hidden_dim, n_heads=n_heads, head_dim=head_dim
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffnn = PositionWiseFeedForwardNet(
            hidden_dim=hidden_dim, ff_dim=ff_dim, dropout_rate=dropout_rate
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, enc_inputs: Tensor, enc_self_attn_mask: Tensor) -> Tensor:
        mh_output = self.self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )
        mh_output = self.layer_norm1(enc_inputs + mh_output)  # Add + norm
        ffnn_output = self.ffnn(mh_output)
        ffnn_output = self.layer_norm2(ffnn_output + mh_output)

        return ffnn_output


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        max_seq_len: int,
        padding_id: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.src_emb = nn.Embedding(input_dim, hidden_dim)
        pe_table = get_positional_encoding_table(max_seq_len, hidden_dim)
        self.pe_emb = nn.Embedding.from_pretrained(pe_table, freeze=True)
        self.padding_id = padding_id
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim, n_heads, head_dim, ff_dim, dropout_rate=dropout_rate
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_input: Tensor):
        """encoder input을 입력받아, pe vector를 생성하고,
        input과 pe를 더해 combined embedding으로 레이어 개수만큼 연산한다.

        Args:
            enc_input (Tensor): (batch_size, max_seq_len, hidden_dim)

        Returns:
            _type_: _description_
        """
        position = get_position(enc_input)
        combined_emb = self.src_emb(enc_input) + self.pe_emb(position)
        padding_mask = get_padding_mask(enc_input, self.padding_id)

        combined_emb
        for layer in self.layers:
            combined_emb = layer(combined_emb, padding_mask)

        return combined_emb


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout_rate: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout_rate,
        )
        self.enc_dec_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout_rate,
        )
        self.ffnn = PositionWiseFeedForwardNet(
            hidden_dim=hidden_dim, ff_dim=ff_dim, dropout_rate=dropout_rate
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.layer_norm3 = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(
        self,
        enc_output: Tensor,
        dec_input: Tensor,
        dec_self_attn_mask: Tensor,
        dec_enc_padding_mask,
    ) -> Tensor:
        output = self.masked_self_attn(
            dec_input, dec_input, dec_input, dec_self_attn_mask
        )
        output = self.layer_norm1(dec_input)
        output = self.enc_dec_attn(
            query=output,
            key=enc_output,
            value=enc_output,
            attn_mask=dec_enc_padding_mask,
        )
        output = self.layer_norm2(output)
        output = self.ffnn(output)
        output = self.layer_norm3(output)

        return output


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        max_seq_len: int,
        padding_id: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.src_emb = nn.Embedding(input_dim, hidden_dim)
        pe_table = get_positional_encoding_table(max_seq_len, hidden_dim)
        self.pos_emb = nn.Embedding.from_pretrained(pe_table, freeze=True)
        self.padding_id = padding_id
        self.classifier = nn.Linear(hidden_dim, input_dim)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    dropout_rate=dropout_rate,
                )
                for layer in range(n_layers)
            ]
        )

    def forward(self, dec_input: Tensor, enc_output: Tensor) -> Tensor:
        position = get_position(dec_input)
        combined_emb = self.src_emb(dec_input) + self.pos_emb(position)
        padding_mask = get_padding_mask(dec_input, self.padding_id)
        look_ahead_mask = get_look_ahead_mask(dec_input)
        dec_self_attn_mask = padding_mask + look_ahead_mask

        dec_enc_padding_mask = get_padding_mask(dec_input, self.padding_id)

        for layer in self.layers:
            combined_emb = layer(
                enc_output, combined_emb, dec_enc_padding_mask, dec_self_attn_mask
            )

        outputs = F.log_softmax(self.classifier(combined_emb), dim=1)

        return outputs


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
        hidden_dim: int,
        max_seq_len: int,
        padding_id: int = 3,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.device = get_device()
        self.encoder = Encoder(
            input_dim=enc_d_input,
            hidden_dim=hidden_dim,
            n_layers=enc_layers,
            n_heads=enc_heads,
            head_dim=enc_head_dim,
            ff_dim=enc_ff_dim,
            max_seq_len=max_seq_len,
            padding_id=padding_id,
            dropout_rate=dropout_rate,
        ).to(self.device)
        self.decoder = Decoder(
            input_dim=dec_d_input,
            hidden_dim=hidden_dim,
            n_layers=dec_layers,
            n_heads=dec_heads,
            head_dim=dec_head_dim,
            ff_dim=dec_ff_dim,
            max_seq_len=max_seq_len,
            padding_id=padding_id,
            dropout_rate=dropout_rate,
        ).to(self.device)

    def forward(self, enc_input: Tensor, dec_input: Tensor):
        enc_outputs = self.encoder(enc_input)
        dec_outputs = self.decoder(dec_input, enc_outputs)

        return dec_outputs


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
    )  # transformer에서 sprt(d_momel)로 나눠주기 때문에 해주는 듯함. (수식에 있음)

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
