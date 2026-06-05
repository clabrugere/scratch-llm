import torch
from torch import Tensor


class KVCache:
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_head: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_head = num_head
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.cache = torch.empty(
            (self.num_layers, 2, self.batch_size, self.num_head, self.max_seq_len, self.head_dim),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.current_seq_len = 0

    def update(self, layer_idx: int, key_new: Tensor, value_new: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = key_new.size(2)  # (bs, num_head, seq_len, head_dim)
        end = self.current_seq_len + seq_len

        self.cache[layer_idx, 0, :, :, self.current_seq_len : end, :] = key_new
        self.cache[layer_idx, 1, :, :, self.current_seq_len : end, :] = value_new

        key = self.cache[layer_idx, 0, :, :, :end, :]
        value = self.cache[layer_idx, 1, :, :, :end, :]

        return key, value

    def step(self, seq_len: int) -> None:
        self.current_seq_len += seq_len
