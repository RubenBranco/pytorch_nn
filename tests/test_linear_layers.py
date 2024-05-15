import pytest
import torch
import torch.nn as nn

from pytorch_nn import Identity, Linear, Bilinear, copy_weights


@pytest.fixture
def in_features():
    return 3


@pytest.fixture
def in2_features():
    return 5


@pytest.fixture
def out_features():
    return 2


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def x(in_features, batch_size, seed):
    torch.manual_seed(seed)
    return torch.rand((batch_size, in_features))


@pytest.fixture
def x2(in2_features, batch_size, seed):
    torch.manual_seed(seed)
    return torch.rand((batch_size, in2_features))


def test_identity(x):
    identity_pnn = Identity()
    identity_torch = nn.Identity()

    assert torch.equal(identity_pnn(x), identity_torch(x))


def test_linear(in_features, out_features, x):
    linear_pnn = Linear(in_features, out_features)
    linear_torch = nn.Linear(in_features, out_features)

    copy_weights(linear_pnn, linear_torch)

    assert torch.allclose(linear_pnn(x), linear_torch(x))

    linear_pnn = Linear(in_features, out_features, bias=False)
    linear_torch = nn.Linear(in_features, out_features, bias=False)

    copy_weights(linear_pnn, linear_torch)

    assert torch.allclose(linear_pnn(x), linear_torch(x))


def test_bilinear(in_features, in2_features, out_features, x, x2):
    bilinear_pnn = Bilinear(in_features, in2_features, out_features)
    bilinear_torch = nn.Bilinear(in_features, in2_features, out_features)

    copy_weights(bilinear_pnn, bilinear_torch)

    assert torch.allclose(bilinear_pnn(x, x2), bilinear_torch(x, x2))

    bilinear_pnn = Bilinear(in_features, in2_features, out_features, bias=False)
    bilinear_torch = nn.Bilinear(in_features, in2_features, out_features, bias=False)

    copy_weights(bilinear_pnn, bilinear_torch)

    assert torch.allclose(bilinear_pnn(x, x2), bilinear_torch(x, x2))
