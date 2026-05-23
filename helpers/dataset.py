import torch
from torch import Tensor
from torch.utils.data import Dataset

from model.tokenizer import BPETokenizer


class NextTokenPredictionDataset(Dataset):
    def __init__(self, input_file: str, seq_len: int, tokenizer: BPETokenizer, eos_id: int | None = None) -> None:
        super().__init__()
        self.seq_len = seq_len

        # load data in memory
        data = []
        with open(input_file) as f:
            for line in f:
                data.extend(tokenizer.encode(line.strip()))
                if eos_id is not None:
                    data.append(eos_id)

        self.data = torch.as_tensor(data, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx : idx + self.seq_len], self.data[idx + 1 : idx + self.seq_len + 1]
