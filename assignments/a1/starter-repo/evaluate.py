from argparse import ArgumentParser
import sys

import numpy as np
import pandas as pd


# Argument parsing:

parser = ArgumentParser()
parser.add_argument("-p", "--predict", dest = "predict_path",
    required = True, help = "path to your model's predicted labels file")
parser.add_argument("-d", "--development", dest = "dev_path",
    required = True, help = "path to the development labels file")
args = parser.parse_args()


# Load predicted and dev CSV files:

predict = pd.read_csv(args.predict_path).set_index("id")
dev = pd.read_csv(args.dev_path).set_index("id")

label_column = "label" if "sst" in args.predict_path else "newsgroup"
accuracy = (dev[label_column] == predict[label_column]).mean()
print("Accuracy: ", accuracy)
