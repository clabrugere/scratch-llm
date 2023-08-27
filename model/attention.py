import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_hidden: int = 1,
        dropout: float = 0.0,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        # the activation function GLU (for gated linear unit) is defined as GLU(a,b)=a⊗σ(b)
        # where a is the first half of the input matrices and bb is the second half.
        self._layers = nn.Sequential()
        for _ in range(num_hidden):
            self._layers.append(nn.Linear(dim_in, dim_hidden, device=device))
            self._layers.append(nn.LayerNorm(dim_hidden, device=device))
            self._layers.append(nn.GLU())
            self._layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


class SelfAttention(nn.Module):
    def __init__(
        self, dim_in: int, dim_k: int, dim_v: int, causal=True, device: str = "cuda"
    ) -> None:
        super().__init__()

        self.dim_k = dim_k
        self.causal = causal

        # Query, Key and Value projections
        self.Q = nn.Linear(dim_in, dim_k, bias=False, device=device)
        self.K = nn.Linear(dim_in, dim_k, bias=False, device=device)
        self.V = nn.Linear(dim_in, dim_v, bias=False, device=device)

        # build the causal mask, masking upper triangular part of attention scores
        self.causal_mask = torch.triu(torch.ones(dim_k, dim_k, device=device)).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.Q(x)  # (bs, seq_len, dim_q)
        keys = self.K(x)  # (bs, seq_len, dim_q)
        values = self.V(x)  # (bs, seq_len, dim_v)

        # compute the correlation between a query q_i and all the keys, for very q_i
        attn_scores = queries @ torch.transpose(keys, 2, 1)  # (bs, seq_len, seq_len)

        # fill the upper triangular part of the attention scores with -inf to inhibit them in the softmax
        if self.causal:
            attn_scores.masked_fill_(self.causal_mask[None, ...], float("-inf"))

        attn_scores = torch.softmax(attn_scores / (self.dim_k**0.5), dim=-1)
        out = attn_scores @ values  # (bs, seq_len, dim_v)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_heads: int,
        dim_k: int,
        dim_out: int,
        causal=True,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        # runs multiple self-attention blocks allowing to attent to different parts of the input sequence
        # and projects to the required dimension using a linear map
        dim_head = dim_in // num_heads
        self.attn_heads = nn.ModuleList(
            [
                SelfAttention(dim_head, dim_k, dim_k, causal=causal, device=device)
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(num_heads * dim_k, dim_out, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for head in self.attn_heads:
            out.append(head(x))

        out = torch.cat(out, dim=-1)
        out = self.projection(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_heads: int,
        dim_k: int,
        causal: bool = True,
        dropout: float = 0.0,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.multihead_attn = MultiHeadAttention(
            dim_in, num_heads, dim_k, dim_in, causal=causal, device=device
        )
        self.layer_norm_1 = nn.LayerNorm(dim_in, device=device)
        self.layer_norm_2 = nn.LayerNorm(dim_in, device=device)
        self.feed_forward = FeedForward(dim_in, dim_in, dropout=dropout, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.multihead_attn(x) + x
        out = self.layer_norm_1(out)
        out = self.feed_forward(out) + out
        out = self.layer_norm_2(out)

        return out
