import torch
from torch import nn


# TODO: implement the transformer encoder
# TODO: implement the positional encoding scheme
# TODO: implement the autoregressive generation
class LLM(nn.Module):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor):
        pass

    @torch.inference_mode()
    def generate(self, inputs):
        pass
