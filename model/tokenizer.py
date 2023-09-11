import os
from torch import Tensor, IntTensor
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


__model_types = ["unigram", "bpe", "word", "char"]


class Tokenizer:
    def __init__(self, path: str = None) -> None:
        self.sp = SentencePieceProcessor()
        self.sp.Load(model_file=path)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    def encode(
        self,
        input: str,
        beg_of_string: bool = False,
        end_of_string: bool = False,
        pad_seq: bool = False,
        seq_len: int = None,
    ) -> Tensor:
        out = self.sp.EncodeAsIds(input, add_bos=beg_of_string, add_eos=end_of_string)

        if pad_seq and len(out) < seq_len:
            out = [*[self.pad_id] * (seq_len - len(out)), *out]

        return IntTensor(out)

    def decode(self, input: Tensor) -> str:
        out = "".join(self.sp.Decode(input.tolist()))

        return out


def train_tokenizer(
    input_file: str, vocab_size: int, output_path: str, pad_id=0, unk_id=1, bod_id=2, eos_id=3, model_type="unigram"
) -> None:
    assert model_type in __model_types, f"Got invalid model_type argument: {model_type}"
    SentencePieceTrainer.Train(
        input=input_file,
        vocab_size=vocab_size,
        model_type=model_type,
        model_prefix=os.path.splitext(output_path)[0],
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bod_id,
        eos_id=eos_id,
    )
