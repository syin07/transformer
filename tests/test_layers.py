import torch, pytest
import torch.nn as nn
from nn_basics.layers import Linear, Embedding, LayerNorm

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

    batch_size = 8
    seq_len = 8
    x = torch.randn(batch_size,seq_len,nin)

    actual = my_layer(x)
    expected = torch_layer(x)

    assert torch.allclose(my_layer(x), torch_layer(x))

    # also test the gradients of the weights matrix
    actual.sum().backward()
    expected.sum().backward()

    assert torch.allclose(my_layer.W.grad, torch_layer.weight.grad)
    if bias:
        assert torch.allclose(my_layer.B.grad, torch_layer.bias.grad)

@pytest.mark.parametrize(
    "num_embeddings, embedding_dim",
    [
        (3, 5),
        (8, 10),
    ]
)
def test_embedding_against_torch(num_embeddings, embedding_dim):
    torch.manual_seed(0)
    my_embedding = Embedding(num_embeddings, embedding_dim)
    torch_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

    with torch.no_grad():
        torch_embedding.weight = nn.Parameter(my_embedding.W.detach().clone().requires_grad_(True))

    batch_size = 8
    x = torch.randint(low=0, high=num_embeddings, size=(batch_size,num_embeddings,))

    actual = my_embedding(x)
    expected = torch_embedding(x)

    assert torch.allclose(actual, expected)

    actual.sum().backward()
    expected.sum().backward()

    assert torch.allclose(my_embedding.W.grad, torch_embedding.weight.grad)

@pytest.mark.parametrize(
    "dim, bias",
    [
        (4, False),
        (8, True),
    ]
)
def test_layernorm_against_torch(dim, bias):
    torch.manual_seed(0)
    my_layernorm = LayerNorm(dim, bias)
    torch_layernorm = torch.nn.LayerNorm(dim, bias=bias)

    with torch.no_grad():
        torch_layernorm.weight = nn.Parameter(my_layernorm.W.detach().clone().requires_grad_(True))
        if bias:
            torch_layernorm.bias = nn.Parameter(my_layernorm.B.detach().clone().requires_grad_(True))

    batch_size = 8
    seq_len = 8
    x = torch.randn(batch_size, seq_len, dim)

    actual = my_layernorm(x)
    expected = torch_layernorm(x)

    assert torch.allclose(actual, expected, atol=1e-6)

    W = torch.randn(dim, dim)

    (actual@W).sum().backward()
    (expected@W).sum().backward()

    assert torch.allclose(my_layernorm.W.grad, torch_layernorm.weight.grad)
    if bias:
        assert torch.allclose(my_layernorm.B.grad, torch_layernorm.bias.grad)