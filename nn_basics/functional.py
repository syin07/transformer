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
    """
    Calculate the log of the softmax of the input matrix.
    
    Stable computation formula by subtracting the max element of X from each row to avoid very small value inside log().

    Input: torch.tensor()

    Output: torch.tensor()
    """
    shifted_X = X - X.amax(dim=-1, keepdim=True)

    return shifted_X - torch.log(torch.sum(torch.exp(shifted_X), dim=-1, keepdim=True))

def cross_entropy(Y_hat, y):
    """
    Computes cross entropy loss

    Input: torch.tensor() matrix, torch.tensor() vector

    Output: torch.tensor() scalar
    """
    return (-log_softmax(Y_hat)[torch.arange(0, Y_hat.shape[-2]), y]).mean()

def ReLU(X):
    """Computes Recified Linear Unit"""
    res = torch.where(X < 0.0, 0.0, X)
    return res


def GELU(X):
    """Computes Gaussian Error Linear Unit"""
    return X*(1 + torch.erf(X/torch.sqrt(torch.tensor(2))))/2


