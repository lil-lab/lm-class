from typing import List, Any, Tuple
import pandas as pd
import torch
from evaluate import load

def rerank_sentences_for_wer(model: Any, wer_data: List[Any], savepath: str):
    """
    Function to rerank candidate sentences in the HUB dataset. For each set of sentences,
    you must assign each sentence a score in the form of the sentence's acoustic score plus
    the sentence's log probability. You should then save the top scoring sentences in a .csv
    file similar to those found in the results directory.

    Inputs:
        model (Any): An n-gram or Transformer model.
        wer_data (List[Any]): Processed data from the HUB dataset. 
        savepath (str): The path to save the csv file pairing sentence set ids and the top ranked sentences.
    """
    # TODO
    pass

def compute_wer(gt_path, model_path):
    # Load the sentences
    ground_truths = pd.read_csv(gt_path)['sentences'].tolist()
    guesses = pd.read_csv(model_path)['sentences'].tolist()

    # Compute wer
    wer = load("wer")
    wer_value = wer.compute(predictions=guesses, references=ground_truths)
    return wer_value
