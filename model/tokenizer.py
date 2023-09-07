import os
from torch import Tensor, IntTensor
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


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
            out = [*out, *[self.pad_id] * (seq_len - len(out))]

        return IntTensor(out)

    def decode(self, input: Tensor) -> str:
        out = "".join(self.sp.Decode(input.tolist()))

        return out


def train_tokenizer(input_file: str, vocab_size: int, output_path: str) -> None:
    prefix = os.path.splitext(output_path)[0]
    args = f"--input={input_file} --vocab_size={vocab_size} --model_prefix={prefix} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
    SentencePieceTrainer.Train(args)
