import torch, pytest
from nn_basics.functional import softmax

@pytest.mark.parametrize("shape", [(3,), (2, 3,), (3, 2, 3,)])
def test_softmax_matches_torch(shape):
    torch.manual_seed(0)
    X = torch.randn(*shape)

    actual = softmax(X)
    expected = torch.softmax(X, dim=-1)
    
    assert torch.allclose(actual, expected)

@pytest.mark.parametrize("shape", [(3,), (2, 3,), (3, 2, 3,), (4, 4, 4, 4)])
def test_softmax_rows_sum_to_one(shape):
    torch.manual_seed(0)
    X = torch.randn(*shape)

    actual = softmax(X).sum(dim=-1)
    expected = torch.ones(X.shape[:-1])

    assert torch.allclose(actual, expected)

@pytest.mark.parametrize("tensor", [[1000.0], [1000.0, 1001.0, 1002.0], [[1000.0, 1001.0], [-1000.0, -999.0]]])
def test_softmax_large_logits(tensor):
    X = torch.tensor(tensor)

    output = softmax(X)
    expected = torch.softmax(X, dim=-1)

    assert torch.allclose(output, expected)
    assert torch.isfinite(output).all()