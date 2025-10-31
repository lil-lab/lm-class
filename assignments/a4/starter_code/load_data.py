import os
from typing import Optional
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer




def load_csv_data(data_folder, filename):
    """Loading csv (dev/test data).
    """
    data_path = os.path.join(data_folder, filename)
    data = pd.read_csv(data_path)
    return data


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



class ModelContextualSimilarityDataset(Dataset):
    def __init__(self, model_type, x_csv, y_csv):
        # Get tokenizer
        if model_type == "gpt2":
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Compute maximum length in either case
        self.max_len = self.get_max_len(x_csv, tokenizer)

        # Process each input
        self.data = []
        self.process_data(model_type, tokenizer, x_csv, y_csv)

    def __len__(self):
        # the length of all the lists should be the same
        return len(self.data)
    
    
    def get_max_len(self, x_csv, tokenizer):
        # TODO
        raise NotImplementedError
        return max_len

    def process_data(self, model_type, tokenizer, x_csv, y_csv):
        #process and save all the data, so that every getitem call is just indexing into the data and returning one data point.
        # you can define any number of helper functions to do this.
        # make sure the data are processed and added to self.data in the order they appear. 
        ## TODO
        raise NotImplementedError


    def __getitem__(self, idx):
        curr_dict = self.data[idx]

        #curr_dict['input_ids'] is the ids of the context sequence, including anything special tokens automatically added by the tokenizer (e.g., [CLS] and [SEP] tokens for bert) while excluding the <strong> tokens in the input.   
        return curr_dict['input_ids'], curr_dict['attention_mask'], curr_dict['span1'], curr_dict['span2'], \
            curr_dict['word1'], curr_dict['word2']
            

class ModelIsolatedSimilarityDataset(Dataset):

    def __init__(self, model_type, x_csv, y_csv):
        # Get tokenizer
        if model_type == "gpt2":
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Compute maximum length in either case
        self.max_len = self.get_max_len(x_csv, tokenizer)
        
        # Process each input
        self.data = []
        self.process_data(model_type, tokenizer, x_csv, y_csv)
        self.tokenizer=tokenizer

    def __len__(self):
        return len(self.data)

    def get_max_len(self, x_csv, tokenizer):
        # TODO
        raise NotImplementedError
        return max_len

    def process_data(self, model_type, tokenizer, x_csv, y_csv):
        # you can define any number of helper functions to do this.
        # make sure the data are processed and added to self.data in the order they appear. 
        raise NotImplementedError

    def __getitem__(self, idx):
        curr_dict = self.data[idx]
        #curr_dict['word1_ids'] and curr_dict['word2_ids'] are the ids corresponding to word1 and word2 respectively, including anything special tokens automatically added by the tokenizer (e.g., [CLS] and [SEP] tokens for bert). 
        return curr_dict['word1_ids'], curr_dict['word1_mask'], curr_dict['span1'], curr_dict['word2_ids'], \
            curr_dict['word2_mask'], curr_dict['span2'], curr_dict['word1'], curr_dict['word2']




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
        
        - For pretrained models, please complete the two customized dataset classes, ModelContextualSimilarityDataset and ModelIsolatedSimilarityDataset, to process the contextual and isolated data for pretrained models. You can also define more helper functions. For Word2Vec, you can define any customized datasets in anyway you want. 
    Inputs:
        model_type (str): One of BERT or GPT-2.
    
    Outputs: 
        ont_dev_data, cont_test_data, isol_dev_data, isol_test_data
        (we recommend these to be in dataloader format)
    """
    # TODO
    
