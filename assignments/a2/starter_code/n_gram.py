"""
n-gram language model for Assignment 2: Starter code.
"""

import os
import sys
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
from collections import Counter
import numpy as np

from load_data import load_data
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer

def get_args():
    """
    You may freely add new command line arguments to this function.
    """
    parser = argparse.ArgumentParser(description='n-gram model')
    parser.add_argument('-t', '--tokenization_level', type=str, default='character',
                        help="At what level to tokenize the input data")
    parser.add_argument('-n', '--n', type=int, default=1,
                        help="The value of n to use for the n-gram model")

    parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    parser.add_argument('-s', '--num_samples', type=int, default=10,
                        help="How many samples should we get from our model??")
    parser.add_argument('-x', '--max_steps', type=int, default=40,
                        help="What should the maximum output length of our samples be?")

    args = parser.parse_args()
    return args

class NGramLM():
    """
    N-gram language model
    """

    def __init__(self, n: int):
        """
        Initializes the n-gram model. You may add keyword arguments to this function
        to modify the behavior of the n-gram model. The default behavior for unit tests should
        be that of an n-gram model without any label smoothing.

        Important for unit tests: If you add <bos> or <eos> tokens to model inputs, this should 
        be done in data processing, outside of the NGramLM class. 

        Inputs:
            n (int): The value of n to use in the n-gram model
        """
        self.n = n

    def log_probability(self, model_input: List[Any], base=np.e):
        """
        Returns the log-probability of the provided model input.

        Inputs:
            model_input (List[Any]): The list of tokens associated with the input text.
            base (float): The base with which to compute the log-probability
        """
        # TODO
        raise(NotImplementedError)

    def generate(self, num_samples: int, max_steps: int, results_file: str):
        """
        Function for generating text using the n-gram model.

        Inputs:
            num_samples (int): How many samples to generate
            max_steps (int): The maximum length of any sampled output
            results_file (str): Where to save the generated examples
        """
        # TODO
        raise(NotImplementedError)

    def learn(self, training_data: List[List[Any]]):
        """
        Function for learning n-grams from the provided training data. You may
        add keywords to this function as needed, provided that the default behavior
        is that of an n-gram model without any label smoothing.
        
        Inputs:
            training_data (List[List[Any]]): A list of model inputs, which should each be lists
                                             of input tokens
        """
        # TODO
        raise(NotImplementedError)
        
def main():
    # Get key arguments
    args = get_args()

    # Get the data for language-modeling and WER computation
    tokenization_level = args.tokenization_level
    model_type = "n_gram"
    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = load_data(tokenization_level, model_type) # TODO

    # Initialize and "train" the n-gram model
    n = args.n
    model = NGramLM(n)
    model.learn(train_data)

    # Evaluate model perplexity
    val_perplexity = evaluate_perplexity(model, val_data)
    print(f'Model perplexity on the val set: {val_perplexity}')
    dev_perplexity = evaluate_perplexity(model, dev_data)
    print(f'Model perplexity on the dev set: {dev_perplexity}')
    test_perplexity = evaluate_perplexity(model, test_data)
    print(f'Model perplexity on the test set: {test_perplexity}')    

    # Evaluate model WER
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_dev_wer_predictions.csv')
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
    dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_test_wer_predictions.csv')
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)

    # Generate text from the model
    generation_path = os.path.join('generations', f'{experiment_name}_n_gram_generation_examples.pkl')
    num_samples = args.num_samples
    max_steps = args.max_steps
    model.generate(num_samples, max_steps, generation_path)
    

if __name__ == "__main__":
    main()
    
