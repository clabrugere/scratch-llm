import os
import torch
from torch import Tensor
from model.tokenizer import Tokenizer, train_tokenizer


# TODO: tokenizer should probably be independent from the dataset as it needs to be used for inference as well
class NextTokenPredictionDataset:
    def __init__(self, input_file: str, vocab_size: int, context_size: int, rebuild_vocab: bool = True) -> None:
        super().__init__()

        self.input_file = input_file
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.rebuild_vocab = rebuild_vocab

        # build the tokenizer
        self.build_tokenizer()

        # load tokenized data in memory
        with open(self.input_file) as f:
            data = f.read()

        self.data = self.tokenizer.encode(data, beg_of_string=True, end_of_string=True)

    def build_tokenizer(self) -> None:
        tokenizer_model_file = os.path.join(os.path.dirname(self.input_file), "tokenizer.model")
        if not os.path.exists(tokenizer_model_file) or self.rebuild_vocab:
            train_tokenizer(self.input_file, self.vocab_size, tokenizer_model_file)

        self.tokenizer = Tokenizer(tokenizer_model_file)

    def get_batch(self, batch_size: int) -> Tensor:
        # sample random starting index in the data and build a batch from there
        indices = torch.randint(self.data.size(0) - self.context_size, (batch_size,))
        inputs = torch.stack([self.data[i : i + self.context_size] for i in indices], dim=0)
        labels = torch.stack([self.data[i + 1 : i + 1 + self.context_size] for i in indices], dim=0).long()

        return inputs, labels
