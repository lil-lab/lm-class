import pytest
from hypothesis import given
import torch

from multilayer_perceptron import MultilayerPerceptronModel
from .test_strategies import l_ints, m_ints_classes, m_ints_batch_size, l_int_sequence_length


@given(l_ints, m_ints_classes)
def test_initialization(num_classes, vocab_size):
    sample_mlp = MultilayerPerceptronModel(num_classes, vocab_size)


@given(l_ints, m_ints_classes, m_ints_batch_size, l_int_sequence_length)
def test_predict_input_type_and_dimension(num_classes, vocab_size, batch_size, sequence_length):
    model = MultilayerPerceptronModel(num_classes, vocab_size)

    input_tensor = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length), dtype=torch.long)

    try:
        _ = model.predict(input_tensor)
    except Exception as e:
        pytest.fail(f"predict method raised an exception with valid input: {e}")
