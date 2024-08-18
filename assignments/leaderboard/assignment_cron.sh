#!/bin/bash
echo assignment_cron.sh called: `date`
HOME=PATH/TO/HOME
PYTHONPATH=/PATH/TO/PYTHON

cd PATH/TO/cs5740-assignments/leaderboard
conda activate $CONDA_ENV

# Example for running the evaluation script for assignment 1
# The token GITHUB_CLI_TOKEN is the one used for Github Classroom
$PYTHONPATH CS5740_1.py --username GITHUB_USERNAME --token GITHUB_CLI_TOKEN

echo assignment_cron.sh finished: `date`
