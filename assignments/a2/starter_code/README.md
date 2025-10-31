# Assignment 2

## Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment.

Example of virtual environment creation using conda:
```
conda create -n env_name python=3.10
conda activate env_name
python -m pip install -r requirements.txt
```

## Evaluation commands

To generate word pair similarity scores using the embeddings:
```
python similarity.py > prediction.csv
  --embedding1 results/embedding_file_words1.txt
  --embedding2 results/embedding_file_words2.txt
  --words data/isolated_similarity/isolated_dev_x.csv
```

To evaluate the word pair similarity scores against human ratings:
```
python evaluate.py
  --predicted prediction.csv
  --development data/isolated_similarity/isolated_dev_y.csv
```


## Submission

You need to submit your test embedding files. 
Please only submit the embeddings corresponding to the test data, as the embeddings can be quite large.

Ensure that the name of the submission files (in the `results/` subfolder) are:

- `word2vec_isol_test_words{1,2}_embeddings.txt`

Note that the order of lines in the `.txt` files need to be the same as the order in the data file (`isolated_test_x.csv`).
