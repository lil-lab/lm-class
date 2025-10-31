from argparse import ArgumentParser
from utils import compute_metrics

parser = ArgumentParser()
parser.add_argument("-ps", "--predicted_sql", dest = "pred_sql",
    required = True, help = "path to your model's predicted SQL queries")
parser.add_argument("-pr", "--predicted_records", dest = "pred_records",
    required = True, help = "path to the predicted development database records")
parser.add_argument("-ds", "--development_sql", dest = "dev_sql",
    required = True, help = "path to the ground-truth development SQL queries")
parser.add_argument("-dr", "--development_records", dest = "dev_records",
    required = True, help = "path to the ground-truth development database records")

args = parser.parse_args()
_, _, record_f1, _ = compute_metrics(args.dev_sql, args.pred_sql, args.dev_records, args.pred_records)
print("Record F1: ", record_f1)
