from typing import Tuple
from collections import defaultdict
import logging

import torch
from torch import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%m-%y %H:%M:%S"
)


def log(epoch, step, batch_size, metrics, mode="train"):
    metrics_print = " - ".join([f"{m}: {v:.3f}" for m, v in metrics.items()])

    if mode == "train":
        print(f"Epoch {epoch + 1:02d} - batch {step + 1}/{batch_size} -", metrics_print, end="\r")
    if mode == "eval":
        print(f"\n\tEpoch {epoch + 1:02d} -", metrics_print)


def train(
    model: Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    lr: float,
    max_epoch: int,
    device: str = DEVICE,
    log_every: int = 10,
) -> Tuple[Module, defaultdict]:
    metrics_tracker = defaultdict(list)
    val_loss_tracker = defaultdict(list)
    batch_size = dl_train.batch_size

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(max_epoch):
        for i, (x, labels) in enumerate(iter(dl_train)):
            x, labels = x.to(device), labels.to(device)
            logits = model(x)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            metrics_tracker["loss"].append(loss.detach().cpu().item())
            if i % log_every == 0:
                log(epoch, i, batch_size, metrics_tracker)

        val_loss = evaluate(model, dl_val, device)
        val_loss_tracker["val_loss"].append(val_loss)

        log(epoch, i, batch_size, val_loss_tracker)

    return model, metrics_tracker


@torch.inference_mode()
def evaluate(model: Module, dl_val: DataLoader, device: DEVICE) -> float:
    model.eval()
    running_loss = 0.0
    num_steps = 0

    for sequence, labels in dl_val:
        sequence, labels = sequence.to(device), labels.to(device)
        logits = model(sequence)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

        running_loss += loss.cpu().item()
        num_steps += 1

    return loss / num_steps
