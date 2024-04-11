import torch
from torch import Tensor

from model.tokenizer import Tokenizer


class NextTokenPredictionDataset:
    def __init__(self, input_file: str, context_size: int, tokenizer: Tokenizer) -> None:
        super().__init__()

        self.input_file = input_file
        self.context_size = context_size

        # load data in memory
        data = []
        with open(self.input_file) as f:
            for line in f:
                data.append(tokenizer.encode(line, beg_of_string=True, end_of_string=True))

        self.data = torch.cat(data, dim=0)

    def get_batch(self, batch_size: int) -> tuple[Tensor]:
        # sample random starting index in the data and build a batch from there
        indices = torch.randint(self.data.size(0) - self.context_size, (batch_size,))
        inputs = torch.stack([self.data[i : i + self.context_size] for i in indices], dim=0)
        labels = torch.stack([self.data[i + 1 : i + 1 + self.context_size] for i in indices], dim=0).long()

        return inputs, labels
