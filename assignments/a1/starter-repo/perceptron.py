""" Perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function headers.
"""
import os
import sys
import argparse
from typing import Dict, List

from util import evaluate, load_data


class PerceptronModel():
    """ Perceptron model for classification.
    """
    def __init__(self, num_features: int, num_classes: int):
        """ Initializes the model.

        Inputs:
            num_features (int): The number of features.
            num_classes (int): The number of classes.
        """
        # TODO: Implement initialization of this model.
        self.weights: Dict[int, Dict[int, float]] = {}
        pass
    
    def score(self, model_input: Dict, class_id: int):
        """ Compute the score of a class given the input.

        Inputs:
            model_input (features): Input data for an example
            class_id (int): Class id.
        
        Returns:
            The output score.
        """
        # TODO: Implement scoring function.
        pass

    def predict(self, model_input: Dict) -> int:
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example

        Returns:
            The predicted class.    
        """
        # TODO: Implement prediction for an input.
        return None
    
    def update_parameters(self, model_input: Dict, prediction: int, target: int, lr: float) -> None:
        """ Update the model weights of the model using the perceptron update rule.

        Inputs:
            model_input (features): Input data for an example
            prediction: The predicted label.
            target: The true label.
            lr: Learning rate.
        """
        # TODO: Implement the parameter updates.
        pass

    def learn(self, training_data, val_data, num_epochs, lr) -> None:
        """ Perceptron model training.

        Inputs:
            training_data: Suggested type is (list of tuple), where each item can be
                a training example represented as an (input, label) pair or (input, id, label) tuple.
            val_data: Validation data.
            num_epochs: Number of training epochs.
            lr: Learning rate.
        """
        # TODO: Implement the training of this model.
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perceptron model')
    parser.add_argument('-d', '--data', type=str, default='sst2',
                        help='Dataset')
    parser.add_argument('-f', '--features', type=str, default='feature_name', help='Feature type')
    parser.add_argument('-m', '--model', type=str, default='perceptron', help='Model type')
    args = parser.parse_args()

    data_type = args.data
    feature_type = args.features
    model_type = args.model

    train_data, val_data, dev_data, test_data = load_data(data_type, feature_type, model_type)

    # Train the model using the training data.
    model = PerceptronModel()

    print("Training the model...")
    # Note: ensure you have all the inputs to the arguments.
    model.learn(train_data, val_data, num_epochs, lr)

    # Predict on the development set. 
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", f"perceptron_{data_type}_{feature_type}_dev_predictions.csv"))

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    evaluate(model,
             test_data,
             os.path.join("results", f"perceptron_{data_type}_{feature_type}_test_predictions.csv"))
