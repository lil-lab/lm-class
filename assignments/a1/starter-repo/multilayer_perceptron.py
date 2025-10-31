"""Multi-layer perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function giving headers. You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.


We adapt shape suffixes style when working with tensors.
See https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd.

Dimension key:

b: batch size
l: max sequence length
c: number of classes
v: vocabulary size

For example,

feature_b_l means a tensor of shape (b, l) == (batch_size, max_sequence_length).
length_1 means a tensor of shape (1) == (1,).
loss means a tensor of shape (). You can retrieve the loss value with loss.item().
"""

import argparse
import os
from collections import Counter
from pprint import pprint
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results


class Tokenizer:
    # The index of the padding embedding.
    # This is used to pad variable length sequences.
    TOK_PADDING_INDEX = 0
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    def _pre_process_text(self, text: str) -> List[str]:
        # TODO: Implement this! Expected # of lines: 5~10
        raise NotImplementedError

    def __init__(self, data: List[DataPoint], max_vocab_size: int = None):
        corpus = " ".join([d.text for d in data])
        token_freq = Counter(self._pre_process_text(corpus))
        token_freq = token_freq.most_common(max_vocab_size)
        tokens = [t for t, _ in token_freq]
        # offset because padding index is 0
        self.token2id = {t: (i + 1) for i, t in enumerate(tokens)}
        self.token2id["<PAD>"] = Tokenizer.TOK_PADDING_INDEX
        self.id2token = {i: t for t, i in self.token2id.items()}

    def tokenize(self, text: str) -> List[int]:
        # TODO: Implement this! Expected # of lines: 5~10
        raise NotImplementedError


def get_label_mappings(
    data: List[DataPoint],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Reads the labels file and returns the mapping."""
    labels = list(set([d.label for d in data]))
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for index, label in enumerate(labels)}
    return label2id, id2label


class BOWDataset(Dataset):
    def __init__(
        self,
        data: List[DataPoint],
        tokenizer: Tokenizer,
        label2id: Dict[str, int],
        max_length: int = 100,
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a single example as a tuple of torch.Tensors.
        features_l: The tokenized text of example, shaped (max_length,)
        length: The length of the text, shaped ()
        label: The label of the example, shaped ()

        All of have type torch.int64.
        """
        dp: DataPoint = self.data[idx]
        # TODO: Implement this! Expected # of lines: ~20
        raise NotImplementedError


class MultilayerPerceptronModel(nn.Module):
    """Multi-layer perceptron model for classification."""

    def __init__(self, vocab_size: int, num_classes: int, padding_index: int):
        """Initializes the model.

        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.padding_index = padding_index
        # TODO: Implement this!
        raise NotImplementedError

    def forward(
        self, input_features_b_l: torch.Tensor, input_length_b: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Inputs:
            input_features_b_l (tensor): Input data for an example or a batch of examples.
            input_length (tensor): The length of the input data.

        Returns:
            output_b_c: The output of the model.
        """
        # TODO: Implement this!
        raise NotImplementedError


class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, data: BOWDataset) -> List[int]:
        """Predicts a label for an input.

        Inputs:
            model_input (tensor): Input data for an example or a batch of examples.

        Returns:
            The predicted class.

        """
        all_predictions = []
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        # TODO: Implement this!
        raise NotImplementedError

    def evaluate(self, data: BOWDataset) -> float:
        """Evaluates the model on a dataset.

        Inputs:
            data: The dataset to evaluate on.

        Returns:
            The accuracy of the model.
        """
        # TODO: Implement this!
        raise NotImplementedError

    def train(
        self,
        training_data: BOWDataset,
        val_data: BOWDataset,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
    ) -> None:
        """Trains the MLP.

        Inputs:
            training_data: Suggested type for an individual training example is
                an (input, label) pair or (input, id, label) tuple.
                You can also use a dataloader.
            val_data: Validation data.
            optimizer: The optimization method.
            num_epochs: The number of training epochs.
        """
        torch.manual_seed(0)
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
            for inputs_b_l, lengths_b, labels_b in tqdm(dataloader):
                # TODO: Implement this!
                raise NotImplementedError
            per_dp_loss = 0

            self.model.eval()
            val_acc = self.evaluate(val_data)

            print(
                f"Epoch: {epoch + 1:<2} | Loss: {per_dp_loss:.2f} | Val accuracy: {100 * val_acc:.2f}%"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiLayerPerceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="sst2",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=3, help="Number of epochs"
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    args = parser.parse_args()

    num_epochs = args.epochs
    lr = args.learning_rate
    data_type = DataType(args.data)

    train_data, val_data, dev_data, test_data = load_data(data_type)

    tokenizer = Tokenizer(train_data, max_vocab_size=20000)
    label2id, id2label = get_label_mappings(train_data)
    print("Id to label mapping:")
    pprint(id2label)

    max_length = 100
    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length)
    val_ds = BOWDataset(val_data, tokenizer, label2id, max_length)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length)
    test_ds = BOWDataset(test_data, tokenizer, label2id, max_length)

    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=Tokenizer.TOK_PADDING_INDEX,
    )

    trainer = Trainer(model)

    print("Training the model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer.train(train_ds, val_ds, optimizer, num_epochs)

    # Evaluate on dev
    dev_acc = trainer.evaluate(dev_ds)
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Predict on test
    test_preds = trainer.predict(test_ds)
    test_preds = [id2label[pred] for pred in test_preds]
    save_results(
        test_data,
        test_preds,
        os.path.join("results", f"mlp_{args.data}_test_predictions.csv"),
    )
