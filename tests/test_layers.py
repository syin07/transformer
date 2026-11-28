import torch, pytest
import torch.nn as nn
from nn_basics.layers import Linear, Embedding, RMSNorm, GatedMLP

@pytest.mark.parametrize(
        "nin, nout",
        [
            (8, 10),
            (7, 11),
        ]
)
def test_linear_against_torch(nin, nout):
    torch.manual_seed(0)
    my_layer = Linear(nin, nout)
    torch_layer = torch.nn.Linear(nin, nout, bias=False)
    with torch.no_grad():
        torch_layer.weight = nn.Parameter(my_layer.W.detach().clone().requires_grad_(True))

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
    "dim",
    [
        (4),
        (8),
    ]
)
def test_rmsnorm_against_torch(dim):
    torch.manual_seed(0)

    my_rmsnorm = RMSNorm(dim)
    torch_rmsnorm = torch.nn.RMSNorm(dim, eps=1e-5)

    X = torch.randn(8, 8, dim)
    actual = my_rmsnorm(X)
    expected = torch_rmsnorm(X)

    assert torch.allclose(actual, expected)

# test written using claude, manually verified
def test_gatedmlp():
    """GatedMLP computes w2(gelu(w1(x)) * w3(x)), with w1 as the activated
    branch and w3 as the linear gate."""
    torch.manual_seed(0)
    dim, ffn_dim = 2, 3
    mlp = GatedMLP(dim, ffn_dim)

    assert hasattr(mlp, "w1")
    assert hasattr(mlp, "w2")
    assert hasattr(mlp, "w3")
    assert mlp.w1.W.shape == (ffn_dim, dim)
    assert mlp.w2.W.shape == (dim, ffn_dim)
    assert mlp.w3.W.shape == (ffn_dim, dim)

    # All three layers must be registered, or the optimizer will never see them.
    registered = {id(p) for p in mlp.parameters()}
    for layer in (mlp.w1, mlp.w2, mlp.w3):
        assert id(layer.W) in registered

    # Reference on random weights and inputs, built from torch's own GELU.
    # Biases are left out: they are either absent or still zero at init.
    X = torch.randn(2, 4, dim)
    reference = (torch.nn.functional.gelu(X @ mlp.w1.W.T) * (X @ mlp.w3.W.T)) @ mlp.w2.W.T
    actual = mlp(X)
    assert actual.shape == (2, 4, dim)
    assert torch.allclose(actual, reference, atol=1e-6)

    # Hand-computed case. The weights are chosen so every output is a small
    # multiple of gelu(1) or gelu(-1):
    #   gelu(1)  =  1 * Phi(1)  =  0.8413447
    #   gelu(-1) = -1 * Phi(-1) = -0.1586553
    with torch.no_grad():
        mlp.w1.W.copy_(torch.tensor([[1.0, 0.0],
                                     [0.0, 1.0],
                                     [1.0, -1.0]]))
        mlp.w2.W.copy_(torch.tensor([[1.0, 0.0, -1.0],
                                     [0.0, 1.0, 1.0]]))
        mlp.w3.W.copy_(torch.tensor([[1.0, 1.0],
                                     [2.0, 0.0],
                                     [0.0, -1.0]]))

    X = torch.tensor([[0.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]])
    # x = [1, 0]: w1 x = [1, 0, 1],  w3 x = [1, 2, 0]  -> hidden = [gelu(1), 0, 0]
    # x = [0, 1]: w1 x = [0, 1, -1], w3 x = [1, 0, -1] -> hidden = [0, 0, -gelu(-1)]
    # x = [1, 1]: w1 x = [1, 1, 0],  w3 x = [2, 2, -1] -> hidden = [2 gelu(1), 2 gelu(1), 0]
    expected = torch.tensor([[0.0, 0.0],
                             [0.8413447460685429, 0.0],
                             [-0.15865525393145707, 0.15865525393145707],
                             [1.6826894921370859, 1.6826894921370859]])
    assert torch.allclose(mlp(X), expected, atol=1e-6)
