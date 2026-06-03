import torch
from torch import Tensor
from torch.utils.data import Dataset

from model.tokenizer import BPETokenizer


class NextTokenPredictionDataset(Dataset):
    """Token-level next-token prediction dataset backed by a trained BPETokenizer."""

    def __init__(self, input_file: str, seq_len: int, tokenizer: BPETokenizer, eos_id: int | None = None) -> None:
        super().__init__()
        self.seq_len = seq_len

        # load data in memory
        data = []
        with open(input_file) as f:
            for line in f:
                stripped = line.strip()
                if stripped == "" and eos_id is not None:
                    data.append(eos_id)
                else:
                    data.extend(tokenizer.encode(stripped + "\n"))

        self.data = torch.as_tensor(data, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx : idx + self.seq_len], self.data[idx + 1 : idx + self.seq_len + 1]
