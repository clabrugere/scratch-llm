import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, Parameter, Sequential

from scratch_llm.model.cache import KVCache, LayerKVCache


class CosinePositionalEncoding(Module):
    def __init__(self, seq_len: int, dim_emb: int, base: int = 10_000) -> None:
        super().__init__()

        indices = torch.arange(0, seq_len, dtype=torch.float)
        scale = 1.0 / (base ** (torch.arange(0, dim_emb, 2, dtype=torch.float) / dim_emb))

        position = torch.zeros(1, 1, seq_len, dim_emb)
        position[:, :, :, 0::2] = torch.sin(indices[None, None, :, None] * scale)
        position[:, :, :, 1::2] = torch.cos(indices[None, None, :, None] * scale)

        self.register_buffer("position", position)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.position  # (bs, num_heads, seq_len, dim_emb)
        return x


class RotaryPositionalEncoding(Module):
    def __init__(self, seq_len: int, dim_emb: int, base: int = 10000) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        indices = torch.arange(0, seq_len, dtype=torch.float)
        scale = 1.0 / (base ** (torch.arange(0, dim_emb, 2, dtype=torch.float) / dim_emb))

        position = torch.outer(indices, scale)
        position = torch.cat((position, position), dim=-1)

        position_cos = torch.cos(position[None, None, :, :])  # (bs, num_heads, seq_len, dim_emb)
        position_sin = torch.sin(position[None, None, :, :])  # (bs, num_heads, seq_len, dim_emb)

        self.register_buffer("position_cos", position_cos)
        self.register_buffer("position_sin", position_sin)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x[..., : self.dim_emb // 2], x[..., self.dim_emb // 2 :]

        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        # x is of shape  (bs, num_heads, seq_len, dim_emb)
        seq_len = x.size(2)
        cos = self.position_cos[:, :, start_pos : start_pos + seq_len, :]
        sin = self.position_sin[:, :, start_pos : start_pos + seq_len, :]

        return (x * cos) + (self._rotate_half(x) * sin)


class RMSNorm(Module):
    # RMSnorm(x_i) = (x_i / RMS(x)) * g_i where RMS(x) = sqrt(1 / n *  sum a_i ** 2)
    def __init__(self, dim_last: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim_last**0.5
        self.gain = Parameter(torch.ones(dim_last), requires_grad=True)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, 2, dim=-1, keepdim=True)
        x = self.scale * self.gain * x / (norm + self.eps)

        return x


class SwiGLU(Module):
    # SwiGLU(x) = (xW + b) ⊗ swish(xZ + c) where W, Z, b, c are learnable params
    def __init__(self, dim_in: int, bias: bool = True) -> None:
        super().__init__()

        self.dim_in = dim_in
        self.linear = Linear(dim_in, 2 * dim_in, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # uses only one weight matrix instead of two
        x = self.linear(x)
        x = F.silu(x[..., : self.dim_in]) * x[..., self.dim_in :]

        return x


class SelfAttention(Module):
    def __init__(self, seq_len: int, dim_emb: int, dim_k: int = None, dim_v: int = None, causal=True) -> None:
        super().__init__()

        self.dim_k = dim_k or dim_emb
        self.dim_v = dim_v or dim_emb
        self.causal = causal

        # Query, Key and Value projections
        self.proj_q = Linear(dim_emb, self.dim_k, bias=False)
        self.proj_k = Linear(dim_emb, self.dim_k, bias=False)
        self.proj_v = Linear(dim_emb, self.dim_v, bias=False)
        self.proj_out = Linear(self.dim_v, self.dim_v, bias=False)

        # Build the causal mask, masking upper triangular part of attention scores
        self.register_buffer("causal_mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())

    def forward(self, x: Tensor, return_scores: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        # projects input to Q, K, V spaces
        q = self.proj_q(x)  # (bs, seq_len, dim_k)
        k = self.proj_k(x)  # (bs, seq_len, dim_k)
        v = self.proj_v(x)  # (bs, seq_len, dim_v)

        # Compute the correlation between a query q_i and all the keys, for every q_i
        attn_scores = q @ torch.transpose(k, 2, 1)  # (bs, seq_len, seq_len)

        # Fill the upper triangular part of the attention scores with -inf to inhibit them in the softmax
        if self.causal:
            attn_scores.masked_fill_(self.causal_mask[None, ...], -torch.inf)

        attn_scores = torch.softmax(attn_scores * self.dim_k**-0.5, dim=-1)  # (bs, seq_len, seq_len)
        out = self.proj_out(attn_scores @ v)  # (bs, seq_len, dim_v)

        if return_scores:
            return out, attn_scores
        else:
            return out


class MultiHeadAttention(Module):
    def __init__(self, max_seq_len: int, num_heads: int, dim_emb: int) -> None:
        super().__init__()

        assert dim_emb % num_heads == 0, "num_heads must be a multiple of dim_emb"

        self.seq_len = max_seq_len
        self.num_heads = num_heads
        self.dim_head = dim_emb // num_heads
        self.dim_emb = dim_emb

        # positional encoding to be applied to query and key projections
        # self.positional_encoding = CosinePositionalEncoding(seq_len, dim_emb // num_heads)
        self.positional_encoding = RotaryPositionalEncoding(max_seq_len, dim_emb // num_heads)

        # Query, Key and Value projections batched into one linear layer
        self.proj_qkv = Linear(dim_emb, 3 * dim_emb, bias=False)
        self.proj_out = Linear(dim_emb, dim_emb, bias=False)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        layer_cache: LayerKVCache | None = None,
        return_scores: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # projects input to Q, K, V spaces
        qkv = self.proj_qkv(x)  # (bs, seq_len, 3 * dim_emb)

        # split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # (bs, seq_len, dim_emb), (bs, seq_len, dim_emb), (bs, seq_len, dim_v)

        # use actual sequence length (may be shorter than self.seq_len during inference)
        bs, seq_len, _ = x.shape

        # split projections between heads -> (bs, num_heads, seq_len, dim_emb)
        q = q.view(bs, seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.view(bs, seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(bs, seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        # apply positional encoding to projections, for each heads
        start_pos = layer_cache.current_seq_len if layer_cache is not None else 0
        q = self.positional_encoding(q, start_pos)  # (bs, num_heads, seq_len, dim_emb)
        k = self.positional_encoding(k, start_pos)  # (bs, num_heads, seq_len, dim_emb)

        # update the KV cache if provided and use the cached keys and values for attention score computation
        if layer_cache is not None:
            k, v = layer_cache.update(k, v)  # (bs, num_heads, seq_len_cache, dim_emb)

        # Compute the correlation between a query q_i and all the keys, for every q_i
        attn_scores = (q @ k.permute(0, 1, 3, 2)) * self.dim_head**-0.5  # (bs, num_heads, seq_len, seq_len)

        # Discard masked tokens from the attention matrix if an attention mask is provided
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, -torch.inf)

        # attention scores are used to build a weighted linear combination of values vectors
        attn_scores = torch.softmax(attn_scores, dim=-1)  # (bs, num_heads, seq_len, seq_len)
        out = attn_scores @ v  # (bs, num_heads, seq_len, dim_v)

        # merge heads
        out = out.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, self.dim_emb)  # (bs, seq_len, dim_v)

        # projects to the output space
        out = self.proj_out(out)  # (bs, seq_len, dim_v)

        return (out, attn_scores) if return_scores else out


class FeedForward(Sequential):
    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = False) -> None:
        super().__init__(
            Linear(dim_in, dim_hidden, bias=bias),
            SwiGLU(dim_hidden),
            Linear(dim_hidden, dim_in, bias=bias),
        )


class TransformerBlock(Module):
    def __init__(
        self,
        max_seq_len: int,
        dim_emb: int,
        attn_num_heads: int,
        ffn_hidden_dim: int,
        ffn_bias: bool = False,
    ) -> None:
        super().__init__()

        self.norm_attn = RMSNorm(dim_emb)
        self.multihead_attn = MultiHeadAttention(max_seq_len, attn_num_heads, dim_emb)
        self.norm_ffn = RMSNorm(dim_emb)
        self.feed_forward = FeedForward(dim_emb, ffn_hidden_dim, bias=ffn_bias)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, layer_cache: LayerKVCache | None = None) -> Tensor:
        x = x + self.multihead_attn(self.norm_attn(x), attn_mask, layer_cache)  # (bs, seq_len, dim_in)
        x = x + self.feed_forward(self.norm_ffn(x))  # (bs, seq_len, dim_in)

        return x  # (bs, seq_len, dim_in)


class TransformerStack(Module):
    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        dim_emb: int,
        attn_num_heads: int,
        ffn_hidden_dim: int,
    ):
        super().__init__()
        self.layers = ModuleList(
            TransformerBlock(max_seq_len, dim_emb, attn_num_heads, ffn_hidden_dim) for _ in range(num_layers)
        )
        # Build the causal mask, masking upper triangular part of attention scores
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        causal_mask.unsqueeze_(0).unsqueeze_(0)  # (1, 1, seq_len, seq_len) to be broadcasted across batches and heads
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Tensor, attn_mask: Tensor = None, kv_cache: KVCache | None = None) -> Tensor:
        query_len = x.size(1)
        if query_len > 1:
            # training or prefill, combine the causal mask with the provided attention mask, if any
            causal_mask = self.causal_mask[:, :, :query_len, :query_len]
            attn_mask = causal_mask if attn_mask is None else (attn_mask | causal_mask)

        if kv_cache is not None:
            assert len(kv_cache) == len(self.layers), (
                "KVCache must have the same number of layers as the TransformerStack"
            )
            for layer, layer_cache in zip(self.layers, kv_cache):
                x = layer(x, attn_mask, layer_cache)
            kv_cache.step(query_len)
        else:
            for layer in self.layers:
                x = layer(x, attn_mask)

        return x
