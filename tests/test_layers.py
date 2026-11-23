import torch, pytest
import torch.nn as nn
from nn_basics.layers import Linear

@pytest.mark.parametrize(
        "nin, nout, bias",
        [
            (8, 10, True),
            (7, 11, False),
        ]
)
def test_linear_against_torch(nin, nout, bias):
    torch.manual_seed(0)
    my_layer = Linear(nin, nout, bias)
    torch_layer = torch.nn.Linear(nin, nout, bias)
    with torch.no_grad():
        torch_layer.weight = nn.Parameter(my_layer.W.detach().clone().requires_grad_(True))
        if bias:
            torch_layer.bias = nn.Parameter(my_layer.B.detach().clone().requires_grad_(True))

    x = torch.randn(5, nin)

    actual = my_layer(x)
    expected = torch_layer(x)

    assert torch.allclose(my_layer(x), torch_layer(x))

    # also test the gradients of the weights matrix
    actual.sum().backward()
    expected.sum().backward()

    assert torch.allclose(my_layer.W, torch_layer.weight)
