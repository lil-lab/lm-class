# Assignment 1

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
python perceptron.py -d newsgroups -f feature_name
python perceptron.py -d sst2 -f feature_name
python multilayer_perceptron.py -d newsgroups -f feature_name
```

### Commands to run unittests

Ensure that your code passes the unittests before submitting it.
The commands can be run from the root directory of the project.
```
pytest tests/test_perceptron.py
pytest tests/test_multilayer_perceptron.py
```

### Submission

Ensure that the name of the submission files (in the `results/` subfolder) are:

- `perceptron_newsgroups_test_predictions.csv`
- `mlp_newsgroups_test_predictions.csv`
- `perceptron_sst2_test_predictions.csv`
- `mlp_sst2_test_predictions.csv`