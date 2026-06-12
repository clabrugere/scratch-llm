import argparse
import dataclasses
import logging

import torch
from torch.optim import AdamW

from scratch_llm.config import LLMConfig, TrainingConfig, get_device
from scratch_llm.helpers.dataset import NextTokenPredictionDataset
from scratch_llm.helpers.trainer import save_checkpoint, train
from scratch_llm.model.llm import LLM
from scratch_llm.model.tokenizer import BPETokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def config_from_args(cls, args):
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small transformer-based language model from scratch.")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--vocab_size", type=int, default=4_000)
    model_group.add_argument("--seq_len", type=int, default=256)
    model_group.add_argument("--dim_emb", type=int, default=128)
    model_group.add_argument("--num_layers", type=int, default=4)
    model_group.add_argument("--num_heads", type=int, default=4)
    model_group.add_argument("--emb_dropout", type=float, default=0.0)
    model_group.add_argument("--ffn_dim_hidden", type=int, default=512)

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--input_file", type=str, required=True)
    train_group.add_argument("--tokenizer_max_training_length", type=int, default=None)
    train_group.add_argument("--device", type=torch.device, default=get_device())
    train_group.add_argument("--num_steps", type=int, default=5_000)
    train_group.add_argument("--batch_size", type=int, default=128)
    train_group.add_argument("--learning_rate", type=float, default=3e-3)
    train_group.add_argument("--weight_decay", type=float, default=1e-2)
    train_group.add_argument("--log_frequency", type=int, default=200)
    train_group.add_argument("--chkpt_dir", type=str, default="checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Training configuration: {args}")

    llm_config = config_from_args(LLMConfig, args)
    train_config = config_from_args(TrainingConfig, args)

    # train tokenizer
    with open(args.input_file, "r") as f:
        text = f.read(train_config.tokenizer_max_training_length)
    tokenizer = BPETokenizer(llm_config.vocab_size)
    eos_id = tokenizer.register_special_token("<eos>")
    tokenizer.train(text)

    # prepare dataset
    ds_train = NextTokenPredictionDataset(args.input_file, llm_config.seq_len, tokenizer, eos_id)
    logger.info(f"Dataset loaded with {len(ds_train)} samples.")

    # initialize model
    model = LLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=llm_config.seq_len,
        dim_emb=llm_config.dim_emb,
        num_layers=llm_config.num_layers,
        attn_num_heads=llm_config.num_heads,
        ffn_hidden_dim=llm_config.ffn_dim_hidden,
        emb_dropout=llm_config.emb_dropout,
    )

    if train_config.device.type == "cuda":
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)

    params_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(p.nelement() * p.element_size() for p in model.buffers())
    size = (params_size + buffer_size) / 1024**2

    logger.info(f"total params: {sum(p.numel() for p in model.parameters()):,d}")
    logger.info(f"model size: {size:.3f}MB")

    # train
    _ = train(
        model=model,
        optimizer=optimizer,
        ds_train=ds_train,
        num_steps=train_config.num_steps,
        batch_size=train_config.batch_size,
        device=train_config.device,
        log_every=train_config.log_frequency,
    )

    # save final checkpoint
    final_chkpt_path = f"{args.chkpt_dir}/final_checkpoint.pt"
    logger.info(f"Saving final checkpoint to {final_chkpt_path}")
    save_checkpoint(model, optimizer, tokenizer, args.chkpt_dir, train_config.num_steps)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
