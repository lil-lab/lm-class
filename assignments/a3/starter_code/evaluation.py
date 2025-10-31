from argparse import ArgumentParser
import sys

from wer import compute_wer

# Argument parsing:

parser = ArgumentParser()
parser.add_argument("-p", "--predict", dest = "predict_path",
    required = True, help = "path to your model's predicted labels file")
parser.add_argument("-d", "--development", dest = "dev_path",
    required = True, help = "path to the development labels file")
args = parser.parse_args()

wer = compute_wer(args.dev_path, args.predict_path)
print("WER: ", wer)
