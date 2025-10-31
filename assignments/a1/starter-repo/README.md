# Commands

## Virtual environment creation

It's highly recommended to use a virtual environment for this assignment.

Virtual environment creation (you may also use venv):

```{sh}
conda create -n cs5740a1_310 python=3.10
conda activate cs5740a1_310
python -m pip install -r requirements.txt
```

## Train and predict commands

Example command for the original code (subject to change, if additional arguments are added):

```{sh}
python perceptron.py -d newsgroups -f bow
python perceptron.py -d sst2 -f bow
python multilayer_perceptron.py -d newsgroups
```

## Commands to run unittests

It's recommended to ensure that your code passes the unittests before submitting it.
The commands can be run from the root directory of the project.

```{sh}
pytest
pytest tests/test_perceptron.py
pytest tests/test_multilayer_perceptron.py
```

Please do NOT commit any code that changes the following files and directories:

tests/
.github/
pytest.ini

Otherwise, your submission may be flagged by GitHub Classroom autograder.

Please DO commit your output labels in results/ following the same name and content format. Our leaderboard periodically pulls your outputs and computes accuracy against hidden test labels. <https://github.com/cornell-cs5740-sp25/leaderboards/>
