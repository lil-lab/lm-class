import os, argparse
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

from load_data import load_data_pretrained_models
from utils import get_similarity_scores, compute_spearman_correlation

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PretrainedEmbeddingModel(nn.Module):
    """
    Pretrained model to extract embeddings from.
    """
    def __init__(self, model: str, layers: str, merge_strategy: str, layer_merging: str):
        """
        Initializes GPT-2 or BERT as an instance variable. You may alter the function header or
        add new keyword arguments as needed. The header and variable descriptions are only intended 
        to guide you.

        You should be loading GPT-2 and BERT from Huggingface transformers. You are restricted to the base
        models for this assignment, meaning that you should use the following strings:
            For GPT-2: "gpt2"
            For BERT: "bert-base-uncased"

        Inputs:
            model (str): Which model to load (GPT-2 or BERT)
            layers (str): The hidden states of which layers should we use? 
            merge_stragegy (str): How do we merge subwords when a word is split into multiple subwords?
            layer_strategy (str): If we use multiple layers, how do we combine them?
        """
        # TODO
        raise(NotImplementedError)

    def extract_embedding_from_outputs(self, model_output, word_span):
        """
        Extracts the embedding corresponding to the input word span according to whatever
        strategy the model has been initialized with.

        This function is only here to guide you. You may choose to use or modify this
        function if you wish.

        As a general tip, looking carefully at what models you load from transformers output
        and how what they output align with different portions of the model source code (accessible
        from the model pages on Huggingface) will be very helpful.

        Input:
            model_output (Any): The output of the model following a forward pass. 
            word_span (torch.LongTensor): A minibatch of word spans, where each span contains
                                          the start and end position of the word in the tokenized model input.
        """
        pass

    def extract_isolated(self, isolated_data: Any):
        """
        Extracts word embeddings for isolated word pairs. You are free to approach this function however
        you would like.

        Inputs:
            isolated_data (Any): A dataset containing processed data for isolated word pairs. Recommended
                                 to be in a Dataloader format.

        Returns:
            word1_embeds (List): A list of embeddings for the first words of the word pairs in the dataset, 
                                 in the order they appear.
            word2_embeds (List): A list of embeddings for the second words of the word pairs in the dataset, 
                                 in the order they appear.
        """
        # TODO
        raise(NotImplementedError)

    def extract_contextual(self, contextual_data: Any):
        """
        Extracts word embeddings for contextual word pairs. You are free to approach this function however
        you would like.

        Inputs:
            contextual_data (Any): A dataset containing processed data for contextual word pairs. Recommended
                                 to be in a Dataloader format.

        Returns:
            word1_embeds (List): A list of embeddings for the first words of the word pairs in the dataset, 
                                 in the order they appear.
            word2_embeds (List): A list of embeddings for the second words of the word pairs in the dataset, 
                                 in the order they appear.
        """
        # TODO
        raise(NotImplementedError)

def get_args():
    """
    You may freely add new command line arguments to this function, or change them.
    """
    parser = argparse.ArgumentParser(description='word2vec model')
    parser.add_argument('-m', '--model_type', type='str', choices=['gpt2', 'bert'],
                        help='Which pretrained model will we use?')
    
    parser.add_argument('-l', '--layers', type=str, default='12',
                        help="The hidden dimension outputs of which layers will we use?")
    parser.add_argument('-sm', '--subword_merging', type=str, 
                        help="How do we merge subwords?")
    parser.add_argument('-lm', '--layer_merging', type=str, 
                        help="How do we merge layers, if we do this at all?")

    parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model_type = args.model_type
    layers = args.layers
    merge_strategy = args.subword_merging
    layer_merging = args.layer_merging
    experiment_name = args.experiment_name

    # Load data
    cont_dev_data, cont_test_data, isol_dev_data, isol_test_data = load_data_pretrained_models(model_type)

    # Load model
    model = PretrainedEmbeddingModel(model_type, layers, merge_strategy, layer_merging)
    model.to(DEVICE)

    # Note: The following code is a template, you can choose to use it or create your 
    # own evaluation pipeline. You are free to change the code as you see fit.

    isol_dev_embeds_word1, isol_dev_embeds_word2 = model.extract_isolated(isol_dev_data)
    isol_test_embeds_word1, isol_test_embeds_word2 = model.extract_isolated(isol_test_data) 
    cont_dev_embeds_word1, cont_dev_embeds_word2 = model.extract_contextual(cont_dev_data)
    cont_test_embeds_word1, cont_test_embeds_word2 = model.extract_contextual(cont_test_data) 

    # Save the embeddings to text file

    # Compute word pair similarity scores using your embedding
    isol_dev_sim_scores = get_similarity_scores(isol_dev_embeds_word1, isol_dev_embeds_word2)
    isol_test_sim_scores = get_similarity_scores(isol_test_embeds_word1, isol_test_embeds_word2)
    cont_dev_sim_scores = get_similarity_scores(cont_dev_embeds_word1, cont_dev_embeds_word2)
    cont_test_sim_scores = get_similarity_scores(cont_test_embeds_word1, cont_test_embeds_word2)

    # Evaluate your similarity scores against human ratings
    isol_dev_corr = compute_spearman_correlation(isol_dev_sim_scores, isol_dev_labels)
    isol_test_corr = compute_spearman_correlation(isol_test_sim_scores, isol_test_labels)
    cont_dev_corr = compute_spearman_correlation(cont_dev_sim_scores, cont_dev_labels)
    cont_test_corr = compute_spearman_correlation(cont_test_sim_scores, cont_test_labels)

    print("\n\n\nEvaluating on: isolated word pairs")
    print("Correlation score on dev set:", isol_dev_corr)
    print("Correlation score on test set:", isol_test_corr)

    print("\n\n\nEvaluating on: contextual word pairs")
    print("Correlation score on dev set:", cont_dev_corr)
    print("Correlation score on test set:", cont_test_corr)


if __name__ == "__main__":
    main()
