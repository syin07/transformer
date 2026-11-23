import torch
import torch.nn as nn

class Linear(nn.Module):
    """A single layer of neurons."""
    def __init__(self, num_inputs, num_outputs, bias: bool):
        super().__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.randn(num_outputs, num_inputs) / (num_inputs ** 0.5))
        self.B = nn.Parameter(torch.zeros(num_outputs)) if self.bias else None

    def forward(self, x):
        out = x @ self.W.T
        return out + self.B if self.bias else out

class Embedding(nn.Module):
    """An embedding layer."""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        return self.W[x]

class LayerNorm(nn.Module):
    def __init__(self, dim, bias: bool):
        super().__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.ones(dim))
        self.B = nn.Parameter(torch.zeros(dim)) if self.bias else None

    def forward(self, x):
        epsilon = 1e-5
        shifted_x = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(unbiased=False,dim=-1, keepdim=True) + epsilon)
        res = shifted_x * self.W
        return res + self.B if self.bias else res


