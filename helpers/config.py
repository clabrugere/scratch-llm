from dataclasses import dataclass

import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


@dataclass
class LLMConfig:
    vocab_size: int
    seq_len: int
    dim_emb: int
    num_layers: int
    num_heads: int
    emb_dropout: float
    ffn_dim_hidden: int


@dataclass
class TrainingConfig:
    retrain_tokenizer: bool
    device: torch.device
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    log_frequency: int
