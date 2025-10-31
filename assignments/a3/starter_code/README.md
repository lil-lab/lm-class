# Assignment 2

## Example commands

### Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment.

Example of virtual environment creation using conda:
```
conda create -n env_name python=3.10
conda activate env_name
python -m pip install -r requirements.txt
```

### Train and predict commands

Example commands (subject to change, just for inspiration):
```
python n_gram.py --n=3 --experiment_name=trigram --num_samples=100
python transformer.py --num_layers=4 --hidden_dim=256 --experiment_name=transformer
```

### Commands to run unittests

Ensure that your code passes the unittests before submitting it.
The commands can be run from the root directory of the project.
```
pytest tests/test_n_gram.py
pytest tests/test_transformer.py
```

### Submission

Ensure that the name of the submission files (in the `results/` subfolder) are:

- `subword_n_gram_test_wer_predictions.csv`
- `character_n_gram_test_wer_predictions.csv`
- `transformer_test_wer_predictions.csv`