from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, Dropout
from torch.nn import Parameter
import torch.nn.functional as F


EPS = torch.finfo(torch.float32).eps


class CosinePositionalEncoding(Module):
    def __init__(self, seq_len: int, dim_emb: int, base: int = 10_000, eps: float = EPS):
        super().__init__()

        indices = torch.arange(0, seq_len, dtype=torch.float)
        div_term = 1 / (base ** (torch.arange(0, dim_emb, 2, dtype=torch.float) / dim_emb) + EPS)

        position = torch.zeros(1, seq_len, dim_emb)
        position[:, :, 0::2] = torch.sin(indices[None, :, None] * div_term)
        position[:, :, 1::2] = torch.cos(indices[None, :, None] * div_term)

        self.register_buffer("position", position)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.position  # (bs, seq_len, dim_in)


class RMSNorm(Module):
    # RMSnorm(x_i) = (x_i / RMS(x)) * g_i + b_i where RMS(x) = sqrt(1 / n *  sum a_i ** 2)
    def __init__(self, dim_last: int, eps: float = EPS):
        super().__init__()

        self.dim_last = dim_last
        self.eps = eps
        self.gain = Parameter(torch.ones(self.dim_last), requires_grad=True)
        self.bias = Parameter(torch.zeros(self.dim_last), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x is of shape (..., dim_last)
        scale = torch.norm(x, dim=-1, keepdim=True) * (self.dim_last**-0.5)
        return (x / (scale + self.eps)) * self.gain + self.bias


class SwiGLU(Module):
    def __init__(self, dim_in: int) -> None:
        # SwiGLU computes the output as SwiGLU(x) = (xW + b) ⊗ swish(xZ + c) where W, Z, b, c are learnable params
        super().__init__()

        self.dim_in = dim_in
        self.linear = Linear(dim_in, 2 * dim_in)

    def forward(self, x: Tensor) -> Tensor:
        # uses only one weight matrix instead of two
        out = self.linear(x)
        return F.silu(out[..., : self.dim_in]) + out[..., self.dim_in :]


class SelfAttention(Module):
    def __init__(self, seq_len: int, dim_emb: int, dim_k: int = None, dim_v: int = None, causal=True) -> None:
        super().__init__()

        if dim_k is None:
            dim_k = dim_emb
        if dim_v is None:
            dim_v = dim_emb

        self.dim_k = dim_k
        self.causal = causal

        # Query, Key and Value projections
        self.projection_query = Linear(dim_emb, dim_k, bias=False)
        self.projection_key = Linear(dim_emb, dim_k, bias=False)
        self.projection_value = Linear(dim_emb, dim_v, bias=False)
        self.projection_out = Linear(dim_v, dim_v, bias=False)

        # Build the causal mask, masking upper triangular part of attention scores
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Tensor, return_scores: bool = False) -> Tensor | Tuple[Tensor, Tensor]:
        # projects input to Q, K, V spaces
        queries = self.projection_query(x)  # (bs, seq_len, dim_k)
        keys = self.projection_key(x)  # (bs, seq_len, dim_k)
        values = self.projection_value(x)  # (bs, seq_len, dim_v)

        # Compute the correlation between a query q_i and all the keys, for every q_i
        attn_scores = queries @ torch.transpose(keys, 2, 1)  # (bs, seq_len, seq_len)

        # Fill the upper triangular part of the attention scores with -inf to inhibit them in the softmax
        if self.causal:
            m_inf = -torch.finfo(attn_scores.dtype).max
            attn_scores.masked_fill_(self.causal_mask[None, ...], m_inf)

        attn_scores = torch.softmax(attn_scores * self.dim_k**-0.5, dim=-1)  # (bs, seq_len, seq_len)
        out = self.projection_out(attn_scores @ values)  # (bs, seq_len, dim_v)

        if return_scores:
            return out, attn_scores
        else:
            return out


class MultiHeadAttention(Module):
    def __init__(
        self, seq_len: int, num_heads: int, dim_emb: int, dim_k: int = None, dim_v: int = None, causal=True
    ) -> None:
        super().__init__()

        assert dim_emb % num_heads == 0, "num_heads must be a multiple of dim_emb"

        if dim_k is None:
            dim_k = dim_emb
        if dim_v is None:
            dim_v = dim_emb

        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dim_head = dim_emb // num_heads
        self.dim_k = dim_k
        self.causal = causal

        # Query, Key and Value projections
        self.projection_query = Linear(dim_emb, dim_k, bias=False)
        self.projection_key = Linear(dim_emb, dim_k, bias=False)
        self.projection_value = Linear(dim_emb, dim_v, bias=False)
        self.projection_out = Linear(dim_v, dim_v, bias=False)

        # Build the causal mask, masking upper triangular part of attention scores
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Tensor, return_scores: bool = False) -> Tensor | Tuple[Tensor, Tensor]:
        # projects input to Q, K, V spaces
        queries = self.projection_query(x)  # (bs, seq_len, dim_k)
        keys = self.projection_key(x)  # (bs, seq_len, dim_k)
        values = self.projection_value(x)  # (bs, seq_len, dim_v)

        # split projections between heads -> (bs, num_heads, seq_len, dim_k)
        queries = queries.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        keys = keys.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 3, 1)
        values = values.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        # Compute the correlation between a query q_i and all the keys, for every q_i
        attn_scores = queries @ keys  # (bs, num_heads, seq_len, seq_len)

        # Fill the upper triangular part of the attention scores with -inf to inhibit them in the softmax
        if self.causal:
            m_inf = -torch.finfo(attn_scores.dtype).max
            attn_scores.masked_fill_(self.causal_mask[None, None, ...], m_inf)

        # attention scores are used to build a weighted linear combination of values vectors
        attn_scores = torch.softmax(attn_scores * self.dim_k**-0.5, dim=-1)  # (bs, num_heads, seq_len, seq_len)
        out = attn_scores @ values  # (bs, num_heads, seq_len, dim_v)
        out = out.permute(0, 2, 1, 3).reshape(-1, self.seq_len, self.dim_k)  # (bs, seq_len, dim_v)

        # projects to the output space
        out = self.projection_out(out)  # (bs, seq_len, dim_v)

        if return_scores:
            return out, attn_scores
        else:
            return out


class FeedForward(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self._layers = Sequential()
        for _ in range(num_hidden - 1):
            self._layers.append(Linear(dim_in, dim_hidden))
            self._layers.append(RMSNorm(dim_hidden))
            self._layers.append(SwiGLU(dim_hidden))
            self._layers.append(Dropout(dropout))
            dim_in = dim_hidden

        self._layers.append(Linear(dim_in, dim_hidden))

    def forward(self, x: Tensor) -> Tensor:
        return self._layers(x)


class TransformerBlock(Module):
    def __init__(
        self,
        seq_len: int,
        dim_emb: int,
        attn_num_heads: int,
        attn_causal: bool = True,
        ffd_num_hidden: int = 2,
        ffd_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Follows LLama 2 architecture:
        # - positional encoding at every block start
        # - RMS pre-normalization instead of layer normalization
        # - SwiGLU activation for the feedforward
        self.pos_encoding = CosinePositionalEncoding(seq_len, dim_emb)
        self.norm_1 = RMSNorm(dim_emb)
        self.multihead_attn = MultiHeadAttention(seq_len, attn_num_heads, dim_emb, causal=attn_causal)
        self.norm_2 = RMSNorm(dim_emb)
        self.feed_forward = FeedForward(dim_emb, dim_emb, num_hidden=ffd_num_hidden, dropout=ffd_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pos_encoding(x)  # (bs, seq_len, dim_in)
        x = x + self.multihead_attn(self.norm_1(x))  # (bs, seq_len, dim_in)
        x = x + self.feed_forward(self.norm_2(x))  # (bs, seq_len, dim_in)

        return x  # (bs, seq_len, dim_in)
