import torch
import torch.nn as nn

class Linear(nn.Module):
    """A single layer of neurons."""
    def __init__(self, nin, nout, bias: bool):
        super().__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.randn(nout, nin) / (nin ** 0.5))
        self.B = nn.Parameter(torch.zeros(nout)) if self.bias else None


    def forward(self, x):
        out = x @ self.W.T
        return out + self.B if self.bias else out


class Embedding:
    pass

class LayerNorm:
    pass

