import logging
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def lr_scheduler(
    optimizer: Optimizer,
    min_lr_ratio: float,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

    return LambdaLR(optimizer, lr_lambda)


def train(
    model: Module,
    dl_train: DataLoader,
    device: torch.device,
    lr: float,
    max_epochs: int,
    weight_decay: float = 0.0,
    log_every: int = 10,
    grad_clip: float = 1.0,
) -> defaultdict:
    logger.info(f"Training on {device}.")

    metrics_tracker = defaultdict(list)
    amp_dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler(
        optimizer,
        min_lr_ratio=0.1,
        num_training_steps=max_epochs * len(dl_train),
        num_warmup_steps=min(100, max_epochs * len(dl_train) // 200),
    )

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch + 1}/{max_epochs}:")

        model.train()
        running_loss = 0.0

        for step, (inputs, labels) in enumerate(dl_train):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running_loss += loss.item()
            metrics_tracker["train_loss"].append(loss.item())
            if step % log_every == 0 or step == len(dl_train) - 1:
                logger.info(
                    f"step {step + 1}/{len(dl_train)} -"
                    f"loss: {running_loss / (step + 1):.4f} -"
                    f"lr: {scheduler.get_last_lr()[-1]:.4f}"
                )

    return metrics_tracker


@torch.inference_mode()
def evaluate(model: Module, dl_val: DataLoader, device: torch.device) -> float:
    model.eval()
    running_loss = 0.0

    for sequence, labels in dl_val:
        sequence, labels = sequence.to(device), labels.to(device)
        logits = model(sequence)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

        running_loss += loss.item()

    return running_loss / len(dl_val)
