import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Embedding, Linear, Module

from model.cache import KVCache
from model.transformer import RMSNorm, TransformerStack


def sample_top_p(probs: Tensor, threshold: float) -> Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)  # (bs, vocab_size), (bs, vocab_size)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (bs, vocab_size)

    mask = cumulative_probs - sorted_probs > threshold
    sorted_probs[mask] = 0.0  # virtually discard tokens with lower probability
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # rescale to sum to 1.0

    next_token = torch.multinomial(sorted_probs, num_samples=1)
    next_token = torch.gather(sorted_indices, dim=-1, index=next_token)

    return next_token


class LLM(Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        dim_emb: int,
        num_layers: int,
        attn_num_heads: int,
        ffn_hidden_dim: int,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.attn_num_heads = attn_num_heads
        self.head_dim = dim_emb // attn_num_heads

        self.token_embedding = Embedding(vocab_size, dim_emb)
        self.emb_dropout = Dropout(emb_dropout)
        self.transformer = TransformerStack(num_layers, max_seq_len, dim_emb, attn_num_heads, ffn_hidden_dim)
        self.norm = RMSNorm(dim_emb)
        self.classification_head = Linear(dim_emb, vocab_size)

        # https://paperswithcode.com/method/weight-tying
        self.token_embedding.weight = self.classification_head.weight

    def forward(self, x: Tensor, cache: KVCache | None = None) -> Tensor:
        x = self.token_embedding(x)  # (bs, seq_len, dim_emb)
        x = self.emb_dropout(x)  # (bs, seq_len, dim_emb)
        x = self.transformer(x, kv_cache=cache)  # (bs, seq_len, dim_emb)
        x = self.norm(x)  # (bs, seq_len, dim_emb)
        x = self.classification_head(x)  # (bs, seq_len, vocab_size)

        return x  # (bs, seq_len, vocab_size)

    @torch.inference_mode()
    def generate(
        self,
        inputs: Tensor,
        max_seq_len: int,
        stop_tokens: set | None = None,
        temperature: float = 0.8,
        top_p: float = 0.8,
    ) -> Tensor:
        assert inputs.dim() == 2, "Input should be a batch of sequences with shape (batch_size, seq_len)"
        assert inputs.size(0) == 1, "Batch size > 1 is not supported for generation"
        assert inputs.size(1) <= self.max_seq_len, "Input sequence length exceeds maximum sequence length"

        self.eval()
        # build KV cache for autoregressive decoding
        cache = KVCache(
            num_layers=self.num_layers,
            batch_size=inputs.size(0),
            num_head=self.attn_num_heads,
            max_seq_len=self.max_seq_len,
            head_dim=self.head_dim,
            device=inputs.device,
            dtype=next(self.parameters()).dtype,
        )

        max_new_tokens = min(max_seq_len, self.max_seq_len - inputs.size(1))
        for _ in range(max_new_tokens):
            # prefill the cache on the first pass, decode after
            inputs_cond = inputs if cache.current_seq_len == 0 else inputs[:, -1:]

            # get logits for the last step only
            logits = self(inputs_cond, cache)[:, -1, :]  # (bs, vocab_size)

            # rescale logits by temperature and convert to probabilities over the vocabulary
            probs = F.softmax(logits / temperature, dim=-1)  # (bs, vocab_size)

            # sample the next token index using top-p sampling
            next_token = sample_top_p(probs, top_p)  # (bs, 1)

            # stop generation when a stop token is generated
            if stop_tokens is not None and next_token.item() in stop_tokens:
                break

            # append to the sequence being generated
            inputs = torch.cat((inputs, next_token), dim=-1)

        return inputs.squeeze().cpu()
