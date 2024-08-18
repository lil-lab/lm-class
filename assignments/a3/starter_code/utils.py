import numpy as np
from scipy.stats import spearmanr


def get_similarity_scores(embeddings1, embeddings2):
    """
    Function to compute the similarity scores between two sets of embeddings.
    """
    return [round(np.dot(embeddings1[idx], embeddings2[idx]), 6) for idx in range(len(embeddings1))]

def compute_spearman_correlation(similarity_scores, human_scores):
    """
    Function to compute the Spearman correlation between the similarity scores and human scores (labels).
    """
    return round(spearmanr(similarity_scores, human_scores).correlation, 6)