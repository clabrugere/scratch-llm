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
        scale = 1 / (base ** (torch.arange(0, dim_emb, 2, dtype=torch.float) / dim_emb) + EPS)

        position = torch.zeros(1, 1, seq_len, dim_emb)
        position[:, :, :, 0::2] = torch.sin(indices[None, None, :, None] * scale)
        position[:, :, :, 1::2] = torch.cos(indices[None, None, :, None] * scale)

        self.register_buffer("position", position)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.position  # (bs, seq_len, dim_in)


class RMSNorm(Module):
    # RMSnorm(x_i) = (x_i / RMS(x)) * g_i where RMS(x) = sqrt(1 / n *  sum a_i ** 2)
    def __init__(self, dim_last: int, eps: float = EPS):
        super().__init__()

        self.eps = eps
        self.gain = Parameter(torch.ones(dim_last), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True))
        return self.gain * x / (scale + self.eps)


class SwiGLU(Module):
    def __init__(self, dim_in: int, bias: bool = True) -> None:
        # SwiGLU(x) = (xW + b) ⊗ swish(xZ + c) where W, Z, b, c are learnable params
        super().__init__()

        self.dim_in = dim_in
        self.linear = Linear(dim_in, 2 * dim_in, bias=bias)

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
        self.proj_q = Linear(dim_emb, dim_k, bias=False)
        self.proj_k = Linear(dim_emb, dim_k, bias=False)
        self.proj_v = Linear(dim_emb, dim_v, bias=False)
        self.proj_out = Linear(dim_v, dim_v, bias=False)

        # Build the causal mask, masking upper triangular part of attention scores
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Tensor, return_scores: bool = False) -> Tensor | Tuple[Tensor, Tensor]:
        # projects input to Q, K, V spaces
        q = self.proj_q(x)  # (bs, seq_len, dim_k)
        k = self.proj_k(x)  # (bs, seq_len, dim_k)
        v = self.proj_v(x)  # (bs, seq_len, dim_v)

        # Compute the correlation between a query q_i and all the keys, for every q_i
        attn_scores = q @ torch.transpose(k, 2, 1)  # (bs, seq_len, seq_len)

        # Fill the upper triangular part of the attention scores with -inf to inhibit them in the softmax
        if self.causal:
            m_inf = -torch.finfo(attn_scores.dtype).max
            attn_scores.masked_fill_(self.causal_mask[None, ...], m_inf)

        attn_scores = torch.softmax(attn_scores * self.dim_k**-0.5, dim=-1)  # (bs, seq_len, seq_len)
        out = self.proj_out(attn_scores @ v)  # (bs, seq_len, dim_v)

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

        # positional encoding to be applied to query and key projections
        self.positional_encoding = CosinePositionalEncoding(seq_len, dim_emb // num_heads)

        # Query, Key and Value projections
        self.proj_q = Linear(dim_emb, dim_k, bias=False)
        self.proj_k = Linear(dim_emb, dim_k, bias=False)
        self.proj_v = Linear(dim_emb, dim_v, bias=False)
        self.proj_out = Linear(dim_v, dim_v, bias=False)

        # Build the causal mask, masking upper triangular part of attention scores
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Tensor, return_scores: bool = False) -> Tensor | Tuple[Tensor, Tensor]:
        # projects input to Q, K, V spaces
        q = self.proj_q(x)  # (bs, seq_len, dim_k)
        k = self.proj_k(x)  # (bs, seq_len, dim_k)
        v = self.proj_v(x)  # (bs, seq_len, dim_v)

        # split projections between heads -> (bs, num_heads, seq_len, dim_k)
        q = q.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        # apply positional encoding to projections, for each heads
        q = self.positional_encoding(q)  # (bs, num_heads, seq_len, dim_k)
        k = self.positional_encoding(k)  # (bs, num_heads, seq_len, dim_k)

        # Compute the correlation between a query q_i and all the keys, for every q_i
        attn_scores = (q @ k.permute(0, 1, 3, 2)) * self.dim_k**-0.5  # (bs, num_heads, seq_len, seq_len)

        # Fill the upper triangular part of the attention scores with -inf to inhibit them in the softmax
        if self.causal:
            m_inf = -torch.finfo(attn_scores.dtype).max
            attn_scores.masked_fill_(self.causal_mask[None, None, ...], m_inf)

        # attention scores are used to build a weighted linear combination of values vectors
        attn_scores = torch.softmax(attn_scores, dim=-1)  # (bs, num_heads, seq_len, seq_len)
        out = attn_scores @ v  # (bs, num_heads, seq_len, dim_v)

        # merge heads
        out = out.permute(0, 2, 1, 3).contiguous().view(-1, self.seq_len, self.dim_k)  # (bs, seq_len, dim_v)

        # projects to the output space
        out = self.proj_out(out)  # (bs, seq_len, dim_v)

        if return_scores:
            return out, attn_scores
        else:
            return out


class FeedForward(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_hidden,
        bias: bool = False,
        normalize: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self._layers = Sequential()
        for _ in range(num_hidden - 1):
            self._layers.append(Linear(dim_in, dim_hidden, bias=bias))
            if normalize:
                self._layers.append(RMSNorm(dim_hidden))
            self._layers.append(SwiGLU(dim_hidden))
            if dropout > 0.0:
                self._layers.append(Dropout(dropout))
            dim_in = dim_hidden

        self._layers.append(Linear(dim_in, dim_hidden, bias=bias))

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
        ffd_bias: bool = False,
        ffd_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Follows LLama 2 architecture:
        # - positional encoding on every head of the multi-head attention query and keys projections
        # - RMS pre-normalization instead of layer normalization
        # - SwiGLU activation for the feedforward
        self.norm_1 = RMSNorm(dim_emb)
        self.multihead_attn = MultiHeadAttention(seq_len, attn_num_heads, dim_emb, causal=attn_causal)
        self.norm_2 = RMSNorm(dim_emb)
        self.feed_forward = FeedForward(
            dim_emb, dim_emb, num_hidden=ffd_num_hidden, bias=ffd_bias, dropout=ffd_dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.multihead_attn(self.norm_1(x))  # (bs, seq_len, dim_in)
        x = x + self.feed_forward(self.norm_2(x))  # (bs, seq_len, dim_in)

        return x  # (bs, seq_len, dim_in)
