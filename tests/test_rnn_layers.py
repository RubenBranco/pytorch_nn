import pytest
import torch
import torch.nn as nn

from pytorch_nn import RNN, copy_rnn_weights


@pytest.fixture
def in_features():
    return 5


@pytest.fixture
def hidden_size():
    return 128


@pytest.fixture
def num_layers():
    return 2


@pytest.fixture
def sequence_length():
    return 10


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def x_batch_first(batch_size, sequence_length, in_features, seed):
    torch.manual_seed(seed)
    return torch.rand((batch_size, sequence_length, in_features))


@pytest.fixture
def x_not_batch_first(batch_size, sequence_length, in_features, seed):
    torch.manual_seed(seed)
    return torch.rand((sequence_length, batch_size, in_features))


def test_rnn_batch_first(x_batch_first, hidden_size, num_layers):
    in_features = x_batch_first.size(-1)
    rnn_pnn = RNN(in_features, hidden_size, num_layers, batch_first=True)
    rnn_torch = nn.RNN(in_features, hidden_size, num_layers, batch_first=True)
    copy_rnn_weights(rnn_pnn, rnn_torch)
    assert torch.allclose(
        rnn_pnn(x_batch_first, None)[0], rnn_torch(x_batch_first)[0], atol=1e-6
    )


def test_rnn_not_batch_first(x_not_batch_first, hidden_size, num_layers):
    in_features = x_not_batch_first.size(-1)
    rnn_pnn = RNN(in_features, hidden_size, num_layers, batch_first=False)
    rnn_torch = nn.RNN(in_features, hidden_size, num_layers, batch_first=False)
    copy_rnn_weights(rnn_pnn, rnn_torch)
    assert torch.allclose(
        rnn_pnn(x_not_batch_first, None)[0], rnn_torch(x_not_batch_first)[0], atol=1e-6
    )
