def sst2_featurize(train_data, val_data, dev_data, test_data, feature_type):
    """ Featurizes an input for the sst2 domain.

    Inputs:
        train_data: The training data.
        val_data: The validation data.
        dev_data: The development data.
        test_data: The test data.
        feature_type: Type of feature to be used.
    """
    # TODO: Implement featurization of input.
    pass


def sst2_data_loader(train_data_filename: str,
                     train_labels_filename: str,
                     dev_data_filename: str,
                     dev_labels_filename: str,
                     test_data_filename: str,
                     feature_type: str,
                     model_type: str):
    """ Loads the data.

    Inputs:
        train_data_filename: The filename of the training data.
        train_labels_filename: The filename of the training labels.
        dev_data_filename: The filename of the development data.
        dev_labels_filename: The filename of the development labels.
        test_data_filename: The filename of the test data.
        feature_type: The type of features to use.
        model_type: The type of model to use.

    Returns:
        Training, validation, dev, and test data, all represented as a list of (input, label) format.

        Suggested: for test data, put in some dummy value as the label.
    """
    # TODO: Load the data from the text format.

    # TODO: Featurize the input data for all three splits.

    return [], [], [], []