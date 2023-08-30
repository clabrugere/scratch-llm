import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential, Linear, GLU, Dropout
from torch.nn import Parameter


EPS = torch.finfo(torch.float32).eps


class CosinePositionalEncoding(Module):
    def __init__(self, max_seq_len: int, base: int = 1000, eps: float = EPS):
        indices = torch.arange(0, max_seq_len)
        period = 1 / (base ** (2 * indices / max_seq_len) + eps)

        position = torch.where(indices % 2 == 0, torch.sin(period * indices), torch.cos(period * indices))
        self.register_buffer("position", position[None, :, None])

    def forward(self, x: Tensor) -> Tensor:
        return self.position + x  # (bs, seq_len, dim_in)


class RMSNorm(Module):
    # RMSnorm(x_i) = (x_i / RMS(x)) * g_i + b_i where RMS(x) = sqrt(1 / n sum a_i ** 2)
    def __init__(self, dim_last: int, eps: float = EPS):
        super().__init__()
        self.dim_last = dim_last
        self.eps = eps
        self.gain = Parameter(torch.ones(self.dim_last))
        self.bias = Parameter(torch.zeros(self.dim_last))

    def forward(self, x: Tensor) -> Tensor:
        # x is of shape (bs, ..., dim_last)
        scale = torch.norm(x, dim=-1, keepdim=True) * (self.dim_last**-0.5)
        return x / (scale + self.eps) * self.gain + self.bias


class SelfAttention(Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int, causal=True) -> None:
        super().__init__()

        self.dim_k = dim_k
        self.causal = causal

        # Query, Key and Value projections
        self.Q = Linear(dim_in, dim_k, bias=False)
        self.K = Linear(dim_in, dim_k, bias=False)
        self.V = Linear(dim_in, dim_v, bias=False)

        # Build the causal mask, masking upper triangular part of attention scores
        causal_mask = torch.triu(torch.ones(dim_k, dim_k)).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Tensor) -> Tensor:
        queries = self.Q(x)  # (bs, seq_len, dim_q)
        keys = self.K(x)  # (bs, seq_len, dim_q)
        values = self.V(x)  # (bs, seq_len, dim_v)

        # Compute the correlation between a query q_i and all the keys, for every q_i
        attn_scores = queries @ torch.transpose(keys, 2, 1)  # (bs, seq_len, seq_len)

        # Fill the upper triangular part of the attention scores with -inf to inhibit them in the softmax
        if self.causal:
            attn_scores.masked_fill_(self.causal_mask[None, ...], float("-inf"))

        attn_scores = torch.softmax(attn_scores / (self.dim_k**0.5), dim=-1)  # (bs, seq_len, seq_len)
        out = attn_scores @ values  # (bs, seq_len, dim_v)

        return out


class MultiHeadAttention(Module):
    def __init__(
        self,
        dim_in: int,
        num_heads: int,
        dim_k: int,
        dim_out: int,
        causal=True,
    ) -> None:
        super().__init__()

        # Runs multiple self-attention blocks allowing to attend to different parts of the input sequence
        # and projects to the required dimension using a linear map
        dim_head = dim_in // num_heads
        self.attn_heads = ModuleList([SelfAttention(dim_head, dim_k, dim_k, causal=causal) for _ in range(num_heads)])
        self.projection = Linear(num_heads * dim_k, dim_out)

    def forward(self, x: Tensor) -> Tensor:
        out = []
        for head in self.attn_heads:
            out.append(head(x))  # (bs, seq_len, dim_head)

        out = torch.cat(out, dim=-1)  # (bs, seq_len, num_heads * dim_k)
        out = self.projection(out)  # (bs, seq_len, dim_out)

        return out  # (bs, seq_len, dim_out)


class FeedForward(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # the activation function GLU (for gated linear unit) is defined as GLU(a,b)=a⊗σ(b)
        # where a is the first half of the input matrices and b is the second half.
        self._layers = Sequential()
        for _ in range(num_hidden):
            self._layers.append(Linear(dim_in, dim_hidden))
            self._layers.append(RMSNorm(dim_hidden))
            self._layers.append(GLU())
            self._layers.append(Dropout(dropout))
            dim_in = dim_hidden

    def forward(self, x: Tensor) -> Tensor:
        return self._layers(x)


class TransformerBlock(Module):
    def __init__(
        self,
        dim_in: int,
        num_heads: int,
        dim_k: int,
        causal: bool = True,
        ffd_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Follows LLama 2 architecture:
        # - normalization before projections (pre-norm)
        # - RMS normalization instead of layer normalization
        # - positional encoding at every block start
        self.pos_encoding = CosinePositionalEncoding(dim_in)
        self.multihead_attn = MultiHeadAttention(dim_in, num_heads, dim_k, dim_in, causal=causal)
        self.feed_forward = FeedForward(dim_in, dim_in, num_hidden=1, dropout=ffd_dropout)
        self.norm_1 = RMSNorm(dim_in)
        self.norm_2 = RMSNorm(dim_in)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pos_encoding(x)  # (bs, seq_len, dim_in)
        x = self.norm_1(x)  # (bs, seq_len, dim_in)
        x = self.multihead_attn(x) + x  # (bs, seq_len, dim_in)
        x = self.norm_2(x)  # (bs, seq_len, dim_in)
        x = self.feed_forward(x) + x  # (bs, seq_len, dim_in)

        return x  # (bs, seq_len, dim_in)
