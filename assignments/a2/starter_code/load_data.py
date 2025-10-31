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
    return [], [], [], []