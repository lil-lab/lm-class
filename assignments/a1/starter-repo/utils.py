"""Utility functions for both perceptron and MLP models."""

import enum
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import pandas as pd

##### Data utilities #####


class DataType(enum.Enum):
    SST2 = "sst2"
    NEWSGROUPS = "newsgroups"


@dataclass(frozen=True)
class DataPoint:
    id: int
    text: str
    label: str | None


def read_labeled_data(
    data_filename: str, labels_filename: str
) -> List[DataPoint]:
    # TODO: implement this! Expected # of lines: <10
    raise NotImplementedError


def read_unlabeled_data(data_filename: str) -> List[DataPoint]:
    # TODO: implement this! Expected # of lines: <10
    raise NotImplementedError


def load_data(
    data_type: DataType,
) -> Tuple[List[DataPoint], List[DataPoint], List[DataPoint], List[DataPoint]]:
    """Loads the data for the given data type. Returns train, val, dev, test."""
    data_type = data_type.value
    f_train_data = "data/" + data_type + "/train/train_data.csv"
    f_train_labels = "data/" + data_type + "/train/train_labels.csv"
    f_dev_data = "data/" + data_type + "/dev/dev_data.csv"
    f_dev_labels = "data/" + data_type + "/dev/dev_labels.csv"
    f_test_data = "data/" + data_type + "/test/test_data.csv"

    train = read_labeled_data(f_train_data, f_train_labels)
    dev = read_labeled_data(f_dev_data, f_dev_labels)
    test = read_unlabeled_data(f_test_data)

    # Shuffle the training data with a fixed seed.
    random.seed(0)
    random.shuffle(train)

    # Take 5% of train for validation.
    val = train[: int(len(train) * 0.05)]
    train = train[int(len(train) * 0.05) :]
    return train, val, dev, test


##### Evaluation utilities #####


def accuracy(preds: List[Any], targets: List[Any]) -> float:
    assert len(preds) == len(targets), (
        f"len(preds)={len(preds)}, len(targets)={len(targets)}"
    )
    assert len(targets) > 0, f"len(targets)={len(targets)}"
    correct = sum([pred == target for pred, target in zip(preds, targets)])
    return correct / len(preds)


def save_results(
    data: List[DataPoint], predictions: List[Any], results_path: str
) -> None:
    """Saves the predictions to a file.

    Inputs:
        predictions (list of predictions, e.g., string)
        results_path (str): Filename to save predictions to
    """
    ids = [d.id for d in data]
    df = pd.DataFrame({"id": ids, "label": predictions})
    df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
