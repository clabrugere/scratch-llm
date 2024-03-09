import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Embedding, Linear, Module, Sequential

from model.transformer import RMSNorm, TransformerBlock


class LLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        dim_emb: int,
        num_layers: int,
        attn_num_heads: int,
        ffd_hidden_dim: int,
        emb_dropout: float = 0.0,
        ffd_bias: bool = False,
    ) -> None:
        super().__init__()

        self.context_size = context_size
        self.token_embedding = Embedding(vocab_size, dim_emb)
        self.emb_dropout = Dropout(emb_dropout)
        self.transformer = Sequential()

        for _ in range(num_layers):
            self.transformer.append(TransformerBlock(context_size, dim_emb, attn_num_heads, ffd_hidden_dim, ffd_bias))

        self.norm = RMSNorm(dim_emb)
        self.projection_head = Linear(dim_emb, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x)  # (bs, seq_len, dim_emb)
        x = self.emb_dropout(x)  # (bs, seq_len, dim_emb)
        x = self.transformer(x)  # (bs, seq_len, dim_emb)
        x = self.norm(x)  # (bs, seq_len, dim_emb)
        x = self.projection_head(x)  # (bs, seq_len, vocab_size)

        return x  # (bs, seq_len, vocab_size)

    @torch.inference_mode()
    def generate(self, inputs: Tensor, max_seq_len: int, temperature: float = 1.0, top_p: int = None) -> Tensor:
        for _ in range(max_seq_len):
            # make sure the sequence we're generating doesn't exceed model's sequence length
            inputs_cond = inputs if inputs.size(1) <= self.context_size else inputs[:, -self.context_size :]

            # get logits for the last sequence only, and rescale them to get a probability distribution over the vocabulary
            logits = self(inputs_cond)[:, -1, :]  # (bs, vocab_size)

            # TODO: Top-p sampling (nucleus)
            probs = F.softmax(logits / temperature, dim=-1)  # (bs, vocab_size)

            # sample the next token index
            next_token = torch.multinomial(probs, num_samples=1)

            # append to the sequence being generated
            inputs = torch.cat((inputs, next_token), dim=-1)

        return inputs
