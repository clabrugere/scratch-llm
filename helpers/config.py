from dataclasses import dataclass


@dataclass
class LLMConfig:
    vocab_size: int = 1024
    context_size: int = 64
    dim_emb: int = 512
    num_layers: int = 4
    num_heads: int = 16


@dataclass
class TrainingConfig:
    retrain_tokenizer: bool = False
    batch_size: int = 128
    learning_rate: float = 1e-4
    max_steps: int = 1000
    log_frequency: int = 10
