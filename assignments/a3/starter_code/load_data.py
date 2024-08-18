import os
from typing import Optional
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def load_data_word2vec(use_additional_data: Optional[bool] = False):
    """
    Function for loading data for training or using vector representations 
    (or embeddings) of words, for word2vec.
    You may modify the function header and outputs as necessary.

    You can create dedicated Dataset and DataLoader to process the data.

    Inputs:
        use_additional_data (bool): Whether to use additional data for training
    """
    # TODO
    return [], [], [], [], [], []

def load_data_pretrained_models(model_type: str):
    """
    Function for loading and processing the evaluation datasets to be used
    by BERT or GPT-2. You may modify the function header and outputs as necessary.

    As general tips:
        - For the contextual data, you will need to take into account the "<strong>" and "</strong>" tags
          in the context string. Your input to the model should not have these tags.
        - Unlike word2vec, you will need to reason about subword tokenization. Your data processing should
          take into account what span of tokens each target word will map to, so that you may extract the
          embeddings from the correct positions in the input sequence. 

          As a sanity check, you should ensure that the tokens in the spans you extract do in fact minimally match
          your target words at the position they occur. Some things to keep in mind:
              - Your tokenizer's .decode() function takes in a sequence of tokens and outputs the corresponding string.
              - The return_offsets_mapping keyword in the tokenizer call can help in debugging.
              - Your target word may be a substring of another word in the sequence. For a given target, your span needs 
                correspond to the word originally in between by the <strong> and </strong> tags.

    Inputs:
        model_type (str): One of BERT or GPT-2.
    """
    # TODO
    return [], [], [], []
