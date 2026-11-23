import torch, pytest
from nn_basics.functional import softmax, log_softmax, cross_entropy, ReLU, GELU

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

@pytest.mark.parametrize("shape", [(3,), (2, 3,), (3, 2, 3,)])
def test_softmax_gradient_matches_torch(shape):
    torch.manual_seed(0)
    X_actual = torch.randn(*shape, requires_grad=True)
    X_expected = X_actual.detach().clone().requires_grad_(True)

    W = torch.randn(5, shape[-1])

    actual = (softmax(X_actual) @ W.T).sum()
    expected = (torch.softmax(X_expected, dim=-1) @ W.T).sum()

    actual.backward()
    expected.backward()

    assert torch.allclose(X_actual.grad, X_expected.grad)

@pytest.mark.parametrize("tensor", [[1000.0, -1000.0], [[1000.0, 1001.0], [-1000.0, -1001.0]]])
def test_softmax_gradient_matches_torch_extreme_inputs(tensor):
    torch.manual_seed(0)
    X_actual = torch.tensor(tensor, requires_grad=True)
    X_expected = X_actual.detach().clone().requires_grad_(True)

    W = torch.randn(5, X_actual.shape[-1])

    actual = (softmax(X_actual) @ W.T).sum()
    expected = (torch.softmax(X_expected, dim=-1) @ W.T).sum()

    actual.backward()
    expected.backward()

    assert torch.allclose(X_actual.grad, X_expected.grad)

@pytest.mark.parametrize("shape", [(3,), (2, 3), (3, 2, 3), (4, 4, 4, 4)])
def test_log_softmax_matches_torch(shape):
    torch.manual_seed(0)
    X = torch.randn(*shape)

    actual = log_softmax(X)
    expected = torch.nn.functional.log_softmax(X, dim=-1)

    assert torch.allclose(actual, expected)

@pytest.mark.parametrize("tensor", [[1000.0, -1000.0], [[1000.0, 1001.0], [-1000.0, -1001.0]]])
def test_log_softmax_wide_range_of_values_and_large_logits(tensor): 
    X = torch.tensor(tensor)

    actual = log_softmax(X)
    expected = torch.nn.functional.log_softmax(X, dim=-1)

    assert torch.allclose(actual, expected)

@pytest.mark.parametrize("shape", [(3,), (2, 3,), (3, 2, 3,)])
def test_log_softmax_gradient_matches_torch(shape):
    torch.manual_seed(0)
    X_actual = torch.randn(*shape, requires_grad=True)
    X_expected = X_actual.detach().clone().requires_grad_(True)

    W = torch.randn(5, shape[-1])

    actual = (log_softmax(X_actual) @ W.T).sum()
    expected = (torch.log_softmax(X_expected, dim=-1) @ W.T).sum()

    actual.backward()
    expected.backward()

    assert torch.allclose(X_actual.grad, X_expected.grad)

@pytest.mark.parametrize("tensor", [[1000.0, -1000.0], [[1000.0, 1001.0], [-1000.0, -1001.0]]])
def test_log_softmax_gradient_matches_torch_extreme_inputs(tensor):
    torch.manual_seed(0)
    X_actual = torch.tensor(tensor, requires_grad=True)
    X_expected = X_actual.detach().clone().requires_grad_(True)

    W = torch.randn(5, X_actual.shape[-1])

    actual = (log_softmax(X_actual) @ W.T).sum()
    expected = (torch.log_softmax(X_expected, dim=-1) @ W.T).sum()

    actual.backward()
    expected.backward()

    assert torch.allclose(X_actual.grad, X_expected.grad)

@pytest.mark.parametrize("shape", [(3, 5), (10, 10)])
def test_cross_entropy_matches_torch(shape):
    torch.manual_seed(0)
    Y_hat = torch.randn(*shape)
    Y = torch.randint(low=0, high=shape[-1], size=(shape[-2],))

    actual = cross_entropy(Y_hat, Y)
    expected = torch.nn.functional.cross_entropy(Y_hat, Y)

    assert torch.allclose(actual, expected)

@pytest.mark.parametrize("tensor", [[[1000.0, 1001.0], [-1000.0, -1001.0]],[[1000.0, -1000.0], [-1000.0, 1000.0]]])
def test_cross_entropy_matches_torch_extreme_inputs(tensor):
    Y_hat = torch.tensor(tensor)
    Y = torch.tensor([1, 0])

    actual = cross_entropy(Y_hat, Y)
    expected = torch.nn.functional.cross_entropy(Y_hat, Y)

    assert torch.allclose(actual, expected)

    
@pytest.mark.parametrize("shape", [(3, 5), (10, 10)])
def test_cross_entropy_gradient_matches_torch(shape):
    torch.manual_seed(0)
    Y_hat_actual = torch.randn(*shape, requires_grad=True)
    Y_hat_expected = Y_hat_actual.detach().clone().requires_grad_(True)

    Y = torch.randint(low=0, high=shape[-1], size=(shape[-2],))

    actual = cross_entropy(Y_hat_actual, Y)
    expected = torch.nn.functional.cross_entropy(Y_hat_expected, Y)

    actual.backward()
    expected.backward()

    assert torch.allclose(Y_hat_actual.grad, Y_hat_expected.grad)

@pytest.mark.parametrize("tensor", [[[1000.0, 1001.0], [-1000.0, -1001.0]], [[1000.0, -1000.0], [1000.0, -1000.0]]])
def test_cross_entropy_gradient_matches_torch_extreme_inputs(tensor):
    torch.manual_seed(0)
    Y_hat_actual = torch.tensor(tensor, requires_grad=True)
    Y_hat_expected = Y_hat_actual.detach().clone().requires_grad_(True)

    Y = torch.tensor([1, 0])

    actual = cross_entropy(Y_hat_actual, Y)
    expected = torch.nn.functional.cross_entropy(Y_hat_expected, Y)

    actual.backward()
    expected.backward()

    assert torch.allclose(Y_hat_actual.grad, Y_hat_expected.grad)

@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 3)])
def test_ReLU_matches_torch(shape):
    torch.manual_seed(0)
    X = torch.randn(*shape)

    actual = ReLU(X)
    expected = torch.relu(X)

    assert torch.allclose(actual, expected)

@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 3)])
def test_ReLU_gradients_matches_torch(shape):
    torch.manual_seed(0)
    X_actual = torch.randn(*shape, requires_grad=True)
    X_expected = X_actual.detach().clone().requires_grad_(True)

    actual = ReLU(X_actual).sum()
    expected = torch.relu(X_expected).sum()

    actual.backward()
    expected.backward()

    assert torch.allclose(X_actual.grad, X_expected.grad)

@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 3)])
def test_GELU_matches_torch(shape):
    torch.manual_seed(0)
    X = torch.randn(*shape)

    actual = GELU(X)
    expected = torch.nn.functional.gelu(X)

    assert torch.allclose(actual, expected)

@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 3)])
def test_GELU_gradients_matches_torch(shape):
    torch.manual_seed(0)
    X_actual = torch.randn(*shape, requires_grad=True)
    X_expected = X_actual.detach().clone().requires_grad_(True)

    actual = GELU(X_actual).sum()
    expected = torch.nn.functional.gelu(X_expected).sum()

    actual.backward()
    expected.backward()

    assert torch.allclose(X_actual.grad, X_expected.grad)