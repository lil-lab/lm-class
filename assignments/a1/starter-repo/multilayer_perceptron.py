""" Multi-layer perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function headers.
"""
import os
import sys
import argparse

from util import evaluate, load_data
import torch
import torch.nn as nn


class MultilayerPerceptronModel(nn.Module):
    """ Multi-layer perceptron model for classification.
    """
    def __init__(self, num_classes, vocab_size):
        """ Initializes the model.

        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
        """
        # TODO: Implement initialization of this model.
        # Note: You can add new arguments, with a default value specified.
        pass
    
    def predict(self, model_input: torch.Tensor):
        """ Predicts a label for an input.

        Inputs:
            model_input (tensor): Input data for an example or a batch of examples.

        Returns:
            The predicted class.    

        """
        # TODO: Implement prediction for an input.
        return None

    def learn(self, training_data, val_data, loss_fct, optimizer, num_epochs, lr) -> None:
        """ Trains the MLP.

        Inputs:
            training_data: Suggested type for an individual training example is 
                an (input, label) pair or (input, id, label) tuple.
                You can also use a dataloader.
            val_data: Validation data.
            loss_fct: The loss function.
            optimizer: The optimization method.
            num_epochs: The number of training epochs.
        """
        # TODO: Implement the training of this model.
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiLayerPerceptron model')
    parser.add_argument('-d', '--data', type=str, default='sst2',
                        help='Dataset')
    parser.add_argument('-f', '--features', type=str, default='feature_name', help='Feature type')
    parser.add_argument('-m', '--model', type=str, default='mlp', help='Model type')
    args = parser.parse_args()

    data_type = args.data
    feature_type = args.features
    model_type = args.model

    train_data, val_data, dev_data, test_data = load_data(data_type, feature_type, model_type)

    # Train the model using the training data.
    model = MultilayerPerceptronModel()

    print("Training the model...")
    # Note: ensure you have all the inputs to the arguments.
    model.learn(train_data, val_data, loss_fct, optimizer, num_epochs, lr)

    # Predict on the development set. 
    # Note: if you used a dataloader for the dev set, you need to adapt the code accordingly.
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", f"mlp_{data_type}_{feature_type}_dev_predictions.csv"))

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    evaluate(model,
             test_data,
             os.path.join("results", f"mlp_{data_type}_{feature_type}_test_predictions.csv"))
