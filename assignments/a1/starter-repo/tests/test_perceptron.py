from perceptron import DataPointWithFeatures, PerceptronModel

import pytest


def test_init():
    model = PerceptronModel()
    assert len(model.weights) == 0
    assert len(model.labels) == 0


def test_score_null_model_return_zero():
    dp = DataPointWithFeatures(
        id=0,
        text="I love this movie",
        label="1",
        features={"bow/love": 1.0, "bow/movie": 1.0},
    )
    model = PerceptronModel()
    assert model.score(dp, "1") == pytest.approx(0.0)


def test_score():
    dp = DataPointWithFeatures(
        id=0,
        text="I love this movie",
        label="1",
        features={"bow/love": 1.0, "bow/movie": 1.0},
    )
    model = PerceptronModel()
    model.labels = {"1", "0"}
    model.weights.update({"bow/love#1": 0.2, "bow/movie#1": 0.1})
    assert model.score(dp, "1") == pytest.approx(0.3)
    assert model.score(dp, "0") == pytest.approx(0.0)


def test_predict():
    dp = DataPointWithFeatures(
        id=0,
        text="I love this movie",
        label=None,
        features={"bow/love": 1.0, "bow/movie": 1.0},
    )
    model = PerceptronModel()
    model.labels = {"1", "0"}
    model.weights.update({"bow/love#1": 0.2, "bow/movie#1": 0.1})
    assert model.predict(dp) == "1"


def test_predict_const():
    dp = DataPointWithFeatures(
        id=0,
        text="I hate this movie",
        label=None,
        features={"bow/movie": 1.0, "bow/hate": 1.0},
    )
    model = PerceptronModel()
    model.labels = {"1"}
    assert model.predict(dp) == "1"


def test_update_parameters():
    dp = DataPointWithFeatures(
        id=0,
        text="I love this movie",
        label="1",
        features={"bow/love": 1.0, "bow/movie": 1.0},
    )
    model = PerceptronModel()
    model.labels = {"1", "0"}
    model.weights.update({"bow/no#0": 10.0, "bow/no#1": -10.0})

    pred = "0"
    lr = 0.1
    model.update_parameters(dp, pred, lr)

    assert model.weights == {
        "bow/no#1": -10.0,
        "bow/no#0": 10.0,
        "bow/love#1": 0.1,
        "bow/love#0": -0.1,
        "bow/movie#1": 0.1,
        "bow/movie#0": -0.1,
    }


def test_train_noop():
    dp = DataPointWithFeatures(
        id=0,
        text="I love this movie",
        label="1",
        features={"bow/love": 1.0, "bow/movie": 1.0},
    )
    model = PerceptronModel()
    model.labels = {"1", "0"}
    model.weights.update({"bow/love#1": 10.0, "bow/love#0": -10.0})

    assert model.predict(dp) == "1"

    train_data = [dp]
    val_data = []
    num_epochs = 1
    lr = 0.1
    model.train(train_data, val_data, num_epochs, lr)
    assert model.weights["bow/love#1"] == 10.0
    assert model.weights["bow/love#0"] == -10.0
    assert all(
        model.weights[k] == 0.0
        for k in model.weights
        if k not in {"bow/love#1", "bow/love#0"}
    )
    assert model.labels == {"1", "0"}


def test_train():
    dp = DataPointWithFeatures(
        id=0,
        text="I love this movie",
        label="1",
        features={"bow/love": 1.0, "bow/movie": 1.0},
    )
    model = PerceptronModel()
    model.labels = {"1", "0"}
    model.weights.update({"bow/love#1": -10.0, "bow/love#0": 10.0})

    assert model.predict(dp) == "0"

    train_data = [dp]
    val_data = []
    num_epochs = 1
    lr = 0.1
    model.train(train_data, val_data, num_epochs, lr)
    assert model.weights == {
        "bow/love#1": -9.9,
        "bow/love#0": 9.9,
        "bow/movie#1": 0.1,
        "bow/movie#0": -0.1,
    }
    assert model.labels == {"1", "0"}
