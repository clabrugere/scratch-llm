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
    vocab_size: int = 1024
    context_size: int = 128
    dim_emb: int = 512
    num_layers: int = 8
    num_heads: int = 8
    emb_dropout: float = 0.0
    ffd_dim_hidden: int = 4 * 512
    ffd_bias: bool = True


@dataclass
class TrainingConfig:
    retrain_tokenizer: bool = False
    device: torch.device = get_device()
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_steps: int = 1000
    log_frequency: int = 10
