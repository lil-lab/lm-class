from argparse import ArgumentParser

import pandas as pd
from scipy.stats import spearmanr


parser = ArgumentParser()
parser.add_argument("-p", "--predicted", dest = "pred_path",
    required = True, help = "path to your model's predicted labels file")
parser.add_argument("-d", "--development", dest = "dev_path",
    required = True, help = "path to the development labels file")
args = parser.parse_args()


pred = pd.read_csv(args.pred_path, index_col = "id")
dev = pd.read_csv(args.dev_path, index_col = "id")

pred.columns = ["predicted"]
dev.columns = ["actual"]

data = dev.join(pred)

print("Correlation:", spearmanr(data).correlation)
