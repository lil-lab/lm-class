import os, argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

from load_data import load_data_word2vec
from utils import get_similarity_scores, compute_spearman_correlation


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Word2vec(nn.Module):
    """
    Word2vec model (here, skip-gram with negative sampling)

    You may add functions to this class if needed.
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initializes the Word2vec model.
        You may add optional keyword arguments to this function if needed.

        Inputs:
            vocab_size (int): Size of the vocabulary
            embed_dim (int): Dimensionality of the word embeddings
        """
        # TODO
        raise(NotImplementedError)

    def forward(self, input_tokens: torch.Tensor, context_tokens: torch.Tensor, negative_context: Optional[torch.Tensor] = None):
        """
        Forward pass of the model to compute the embeddings.

        Inputs:
            input_tokens (torch.Tensor): Input words (w)
            context_tokens (torch.Tensor): Context words (c)
            negative_context (torch.Tensor): Negative context word
        
        Outputs:
            input_tokens_embeds (torch.Tensor): Embeddings of input words
            context_embeds (torch.Tensor): Embeddings of context words
            negative_embeds (torch.Tensor): Embeddings of negative context words
        """
        # TODO
        raise(NotImplementedError)
    
    def compute_loss(self, input_embeds: torch.Tensor, context_embeds: torch.Tensor, negative_embeds: Optional[torch.Tensor] = None):
        """
        Computes the loss using the embeddings from the forward pass. If negative_embeds is not None,
        it includes the loss from negative sampling.

        Inputs:
            input_embeds (torch.Tensor): Embeddings of input words (w)
            context_embeds (torch.Tensor): Embeddings of context words (c)
            negative_embeds (torch.Tensor): Embeddings of negative context words
        
        Outputs:
            loss (torch.Tensor)
        """
        # TODO
        raise(NotImplementedError)
    
    def pred(self, input_tokens: torch.Tensor):
        """
        Predicts the embeddings of the input tokens.

        Inputs:
            input_tokens (torch.Tensor): Input words
        
        Outputs:
            embeds (torch.Tensor): Embeddings of input words
        """
        # TODO
        raise(NotImplementedError)
    
    def learn(self, train_data, num_epochs: int):
        """
        Training word2vec model.
        You may change the header as you see fit.
        """
        # TODO
        raise(NotImplementedError)


def get_args():
    """
    You may freely add new command line arguments to this function, or change them.
    """
    parser = argparse.ArgumentParser(description='word2vec model')
    parser.add_argument('-a', '--additional_data', action='store_true',
                        help='Include additional data for training')
    
    parser.add_argument('-d', '--embed_dim', type=int, default=300,
                        help='Dimensionality of the word embeddings')
    parser.add_argument('-w', '--window_size', type=int, default=5,
                        help='Size of the context window')
    parser.add_argument('-wt', '--window_type', type=str, default='linear',
                        help='Type of the context window')
    parser.add_argument('-n', '--num_neg_samples', type=int, default=5,
                        help='(For negative sampling) number of negative samples to use')
    parser.add_argument('-ne', '--neg_exponent', type=float, default=0.75,
                        help='(For negative sampling) exponent for negative sampling distribution')
    parser.add_argument('--min-freq', type=int, default=5,
                        help='The minimum frequency of words to consider')

    parser.add_argument('-ep', '--num_epochs', type=int, default=1,
                        help='Number of training epochs')

    parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    additional_data = args.additional_data
    embed_dim = args.embed_dim
    window_size = args.window_size
    window_type = args.window_type
    num_neg_samples = args.num_neg_samples
    neg_exponent = args.neg_exponent
    min_freq = args.min_freq
    num_epochs = args.num_epochs
    experiment_name = args.experiment_name

    # Load data
    train_dataset, train_loader, isol_dev_data, isol_test_data = load_data_word2vec()
    
    vocab_size = train_dataset.vocab_size

    model = Word2vec(vocab_size, embed_dim)
    model = model.to(DEVICE)

    # Could: add the optimizer / scheduler

    # Training
    model.learn(train_loader, num_epochs)

    # Could: save the model

    # Note: The following code is a template, you can choose to use it or create your 
    # own evaluation pipeline. You are free to change the code as you see fit.

    # Compute the embeddings for dev/test words (using the adapted inputs)
    # The inputs given below are just placeholder examples
    isol_dev_embeds_word1 = model.pred(isol_dev_data.word1_ids)
    isol_dev_embeds_word2 = model.pred(isol_dev_data.word2_ids)
    isol_test_embeds_word1 = model.pred(isol_test_data.word1_ids)
    isol_test_embeds_word2 = model.pred(isol_test_data.word2_ids)


    # Could: save the embeddings to text file

    # Compute word pair similarity scores using your embedding
    isol_dev_sim_scores = get_similarity_scores(isol_dev_embeds_word1, isol_dev_embeds_word2)
    isol_test_sim_scores = get_similarity_scores(isol_test_embeds_word1, isol_test_embeds_word2)


    # Evaluate your similarity scores against human ratings
    ## Read the labels

    ## Compute the scores
    isol_dev_corr = compute_spearman_correlation(isol_dev_sim_scores, isol_dev_labels)
    isol_test_corr = compute_spearman_correlation(isol_test_sim_scores, isol_test_labels)


    print("Evaluating on: word pairs in isolation")
    print("Correlation score on dev set:", isol_dev_corr)
    print("Correlation score on test set:", isol_test_corr)
    

if __name__ == "__main__":
    main()
