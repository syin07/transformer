import torch

def softmax(X):
    """
    Calculate the softmax of the input matrix.

    Uses the stable softmax equation to prevent floating point overflow in big inputs.

    Input: torch.tensor()
    
    Output: torch.tensor()
    """
    shifted_X = X - X.amax(dim=-1, keepdim=True)

    return torch.exp(shifted_X)/torch.sum(torch.exp(shifted_X), dim=-1, keepdim=True)

def log_softmax(X):
    pass

def cross_entropy(X):
    pass

def ReLU(X):
    pass

def GELU(X):
    pass


