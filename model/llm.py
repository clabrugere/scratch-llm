import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Embedding, Linear, Module, Sequential

from model.transformer import RMSNorm, TransformerBlock


class LLM(Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        dim_emb: int,
        num_layers: int,
        attn_num_heads: int,
        ffn_hidden_dim: int,
        ffn_bias: bool = False,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.token_embedding = Embedding(vocab_size, dim_emb)
        self.emb_dropout = Dropout(emb_dropout)
        self.transformer = Sequential()

        for _ in range(num_layers):
            self.transformer.append(TransformerBlock(seq_len, dim_emb, attn_num_heads, ffn_hidden_dim, ffn_bias))

        self.norm = RMSNorm(dim_emb)
        self.projection_head = Linear(dim_emb, vocab_size)

        # https://paperswithcode.com/method/weight-tying
        self.token_embedding.weight = self.projection_head.weight

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x)  # (bs, seq_len, dim_emb)
        x = self.emb_dropout(x)  # (bs, seq_len, dim_emb)
        x = self.transformer(x)  # (bs, seq_len, dim_emb)
        x = self.norm(x)  # (bs, seq_len, dim_emb)
        x = self.projection_head(x)  # (bs, seq_len, vocab_size)

        return x  # (bs, seq_len, vocab_size)

    @staticmethod
    def sample_top_p(probs: Tensor, threshold: float) -> Tensor:
        sorted_probs, sorted_indices = torch.sort(probs)  # (bs, vocab_size), (bs, vocab_size)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (bs, vocab_size)

        mask = cumulative_probs < threshold
        sorted_probs[mask] = 0.0  # virtually discard tokens with lower probability
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # rescale to sum to 1.0

        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_indices, dim=-1, index=next_token)

        return next_token

    @torch.inference_mode()
    def generate(self, inputs: Tensor, max_seq_len: int, temperature: float = 0.6, top_p: int = 0.8) -> Tensor:
        for _ in range(max_seq_len):
            # make sure the sequence we're generating doesn't exceed model's sequence length
            inputs_cond = inputs if inputs.size(1) <= self.seq_len else inputs[:, -self.seq_len :]

            # get logits for the last step only, and rescale them to get a probability distribution over the vocabulary
            logits = self(inputs_cond)[:, -1, :]  # (bs, vocab_size)
            probs = F.softmax(logits / temperature, dim=-1)  # (bs, vocab_size)

            # sample the next token index using top-p sampling
            next_token = self.sample_top_p(probs, top_p)  # (bs, 1)

            # append to the sequence being generated
            inputs = torch.cat((inputs, next_token), dim=-1)

        return inputs
