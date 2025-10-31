import pytest
import torch
from multilayer_perceptron import (
    DataPoint,
    MultilayerPerceptronModel,
    Tokenizer,
)


@pytest.mark.parametrize("vocab_size,num_classes", [(1000, 2), (100_000, 10)])
def test_init(vocab_size, num_classes):
    _ = MultilayerPerceptronModel(vocab_size, num_classes, 0)


@pytest.mark.parametrize(
    "vocab_size,num_classes,batch_size,sequence_length",
    [(1000, 2, 16, 1024), (100_000, 10, 64, 512)],
)
def test_predict(
    vocab_size: int, num_classes: int, batch_size: int, sequence_length: int
):
    model = MultilayerPerceptronModel(vocab_size, num_classes, 0)
    batch_size = 2
    sequence_length = 3
    input_features_b_l = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
    )
    input_length_b = torch.randint(
        low=0, high=sequence_length, size=(batch_size,), dtype=torch.long
    )

    output_b_c = model.forward(input_features_b_l, input_length_b)
    assert output_b_c.shape == (batch_size, num_classes)


def test_tokenizer():
    dp = DataPoint(
        id=0,
        text="I love this movie",
        label="1",
    )
    data = [dp]
    tokenizer = Tokenizer(data, max_vocab_size=1000)
    assert len(tokenizer.token2id) == 3
    assert all(t in tokenizer.token2id for t in ("love", "movie"))
    assert len(tokenizer.tokenize("I love this movie")) == 2
