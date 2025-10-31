from argparse import ArgumentParser
import sys

import numpy as np
import pandas as pd


# Argument parsing
parser = ArgumentParser()
parser.add_argument("-e1", "--embedding1", dest = "emb_path1",
    required = True, help = "path to your embedding (of word1)")
parser.add_argument("-e2", "--embedding2", dest = "emb_path2",
    required = True, help = "path to your embedding (of word2)")
parser.add_argument("-w", "--words", dest = "pairs_path",
    required = False, help = "path to dev_x or test_x word pairs, \
    to sanity check that the mapping of word pairs in the embedding files is correct.")
args = parser.parse_args()


def read_embedding(path):
    embeddings = []
    dim = None
    rows = open(path).readlines()
    for i, row in enumerate(rows):
        word, *vector = row.split()
        embeddings.append((word, [float(x) for x in vector]))
        if dim and len(vector) != dim:
            print("Inconsistent embedding dimensions!", file = sys.stderr)
            sys.exit(1)
        dim = len(vector)
    return embeddings, dim


E1, dim1 = read_embedding(args.emb_path1)
E2, dim2 = read_embedding(args.emb_path2)
pairs = pd.read_csv(args.pairs_path)

assert len(E1) == len(E2) == len(pairs)
assert dim1 == dim2

similarity_scores = []
for idx in range(len(pairs)):
    assert (E1[idx][0], E2[idx][0]) == (pairs.word1[idx], pairs.word2[idx])
    sim_score = round(np.dot(E1[idx][1], E2[idx][1]), 6)
    similarity_scores.append(sim_score)
pairs["sim"] = similarity_scores

pairs = pairs.drop([col for col in pairs.columns if col not in ['id', 'sim']], axis=1)

print("Detected a", dim1, "dimension embedding.", file = sys.stderr)
pairs.to_csv(sys.stdout, index = False) # pairs df already has the id column
print("Saved the similarity scores to stdout.", file = sys.stderr)
