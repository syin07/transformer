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
    
    This is a separate method from softmax instead of just doing log(softmax(X)) becuase softmax(X) can produce very small values which then lead to log(softmax(X)) returning very large negative values.

    Stable computation formula by subtracting the max element of X from each row to avoid very small value inside log().

    Input: torch.tensor()

    Output: torch.tensor()
    """
    shifted_X = X - X.amax(dim=-1, keepdim=True)

    return shifted_X - torch.log(torch.sum(torch.exp(shifted_X), dim=-1, keepdim=True))

def cross_entropy(X):
    pass

def ReLU(X):
    pass

def GELU(X):
    pass


