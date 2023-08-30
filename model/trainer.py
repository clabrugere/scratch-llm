from collections import defaultdict
from typing import Tuple
import logging

from torch import Module
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%m-%y %H:%M:%S"
)


def log(epoch, step, batch_size, metrics):
    print(f"Epoch {epoch + 1:02d} - batch {step + 1}/{batch_size} - ", end="\r")


def train(model: Module, dl_train: DataLoader, lr: float, max_epoch: int) -> Module:
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    model.train()
    device = model.device

    for epoch in range(max_epoch):
        for i, (x, labels) in enumerate(iter(dl_train)):
            x, labels = x.to(device), labels.to(device)
            logits = model(x)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model
