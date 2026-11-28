import torch
import torch.nn as nn
from nn_basics.functional import GELU

class Linear(nn.Module):
    """A single layer of neurons."""
    def __init__(self, num_inputs, num_outputs: bool):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_outputs, num_inputs) / (num_inputs ** 0.5))

    def forward(self, x):
        return x @ self.W.T

class Embedding(nn.Module):
    """An embedding layer."""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        return self.W[x]

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.W * (x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))

class GatedMLP(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim

        self.w1 = Linear(dim, ffn_dim)
        self.w2 = Linear(ffn_dim, dim)
        self.w3 = Linear(dim, ffn_dim)

    def forward(self, X):
        return self.w2((GELU(self.w1(X)) * self.w3(X)))


