import pytest
import random
from copy import deepcopy
from hypothesis import given

from perceptron import PerceptronModel
from .test_strategies import m_ints_classes, l_ints

# Simple sanity checks and debugging tests for PerceptronModel


@pytest.mark.perceptron_1
@given(l_ints, m_ints_classes)
def test_initialization(num_features, num_classes):
    sample_model = PerceptronModel(num_features, num_classes)

    assert all(sample_model.weights[class_id][feature_id] == 0.0 
        for class_id in range(num_classes) for feature_id in range(num_features)
    )


@pytest.mark.perceptron_2
@given(l_ints, m_ints_classes)
def test_score(num_features, num_classes):
    sample_model = PerceptronModel(num_features, num_classes)
    # Could include a bias term
    with_bias = len(sample_model.weights[0]) == num_features + 1

    # Case 1
    for class_id in range(num_classes):
        sample_model.weights[class_id] = {feature_id: 1.0 for feature_id in range(num_features)}
        if with_bias:
            sample_model.weights[class_id][num_features] = 0.5
    
    model_input = {feature_id: random.randint(0, 1) for feature_id in range(num_features)}
    for class_id in range(num_classes):
        expected_score = sum(model_input.values()) + (0.5 if with_bias else 0)
        assert sample_model.score(model_input, class_id) == expected_score
    
    # Case 2
    for class_id in range(num_classes):
        sample_model.weights[class_id] = {feature_id: feature_id + 1.0 for feature_id in range(num_features)}
        if with_bias:
            sample_model.weights[class_id][num_features] = 1.0
    
    for class_id in range(num_classes):
        expected_score = sum((feature_id + 1.0) * model_input[feature_id] for feature_id in model_input)
        if with_bias:
            expected_score += 1.0
        assert sample_model.score(model_input, class_id) == expected_score


@pytest.mark.perceptron_3
@given(l_ints, m_ints_classes)
def test_predict(num_features, num_classes):
    sample_model = PerceptronModel(num_features, num_classes)

    # Case 1
    for class_id in range(num_classes):
        sample_model.weights[class_id] = {feature_id: 1.0 for feature_id in range(num_features + 1)}
    
    model_input = {feature_id: random.randint(0, 1) for feature_id in range(num_features)}

    prediction = sample_model.predict(model_input)
    assert 0 <= prediction < num_classes

    # Case 2
    for class_id in range(num_classes):
        sample_model.weights[class_id] = {feature_id: class_id + 1 for feature_id in range(num_features + 1)}

    prediction = sample_model.predict(model_input)
    assert prediction == num_classes - 1


@pytest.mark.perceptron_4
@given(l_ints, m_ints_classes)
def test_update_parameters(num_features, num_classes):
    sample_model = PerceptronModel(num_features, num_classes)
    lr = 0.1
    model_input = {feature_id: random.randint(0, 1) for feature_id in range(num_features)}
    initial_weights = deepcopy(sample_model.weights)

    sample_model.update_parameters(model_input, 0, 0, lr)
    assert sample_model.weights == initial_weights

    sample_model.update_parameters(model_input, 0, 1, lr)
    assert sample_model.weights != initial_weights
    for feature_id in model_input:
        if model_input[feature_id]: # for features in the input
            assert sample_model.weights[0][feature_id] < initial_weights[0][feature_id]
            assert sample_model.weights[1][feature_id] > initial_weights[1][feature_id]