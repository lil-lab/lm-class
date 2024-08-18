Assignment 1 Rubric
===================

Perceptron Features (16pt)
--------------------------
* -1 per
  * Vague description of how a perceptron feature is computed.
  * No mention at all of how or whether word frequency is used
  * No description for bag of words features are provided.
* -2 per
  * No mention at all of how unknown words are processed
* -3 per (cap of -9)
  * Each dataset should have 3 feature sets in addition to bag of words. Remove three points for each absent feature set.
  * What doesn’t count as features:
    * Bag-of-words unigrams
    * Specifying presence/absence or count for n-grams
    * Any n-gram above n > 1 is one feature
    * Most preprocessing (removal of stopwords/punctuation/urls/emails, etc) other than lemmatization/stemming

Experimental setup (8pt)
------------------------
### Perceptron hyperparameters (2pt)
* -1 per missing or nonsense blank

### MLP implementation (4pt)
* -1 per unclear description of an MLP layer (cannot implement in PyTorch without guesswork)
* -1.5 pt if no nonlinearity

### MLP Hyperparameters (2pt)
* -0.5 per missing or nonsense blank 
* -0.5 if they do not use the validation set for stopping


SST and Newsgroup Results and Analyses (14pt each)
--------------------------------------------------
### Quantitative results (5pt)
* -1 per
  * Missing or nonsensical ablations for features described in Section 1
  * Missing or nonsensical ablation experiments for MLP
  * Ablations are better than full model on dev
  * Unclear what the ablations represent
  * Missing test results or reports non-final results (compared to leaderboard) on test set
* +2 per
  * Places at top-3 in the leaderboard for any experiment (can be used once)
    * Once for all the leaderboards. [e.g. for A1, we’ll be giving +2 for 6 people, the 3 at the top of each leaderboard]

### Training loss and validation accuracy plots (3pt)
* -0.5 per
  * Plots are hard to read (e.g., font too small, axis not clear)
* -1.5 per
  * Missing or nonsensical plot

### Qualitative Error Analysis (6pt)
* -1 per detail
  * Unclear description of error class
  * No example shown
* -1 per section
  * No or poor description of error statistics (includes not reporting statistics for both models if model type is both)
  * Error category missing for a model type
* -2 per detail	
  * If less than three error classes are given, two points off for each class missing or invalid (already added to the rubric)

Batching Benchmarking (6pt)
---------------------------
* -0.5 per
  * An entry is missing or nonsensical in the batching benchmark experiment
* -0.25 if missing units (symbolic)

Autograding (10pt)
------------------
* 5 unit tests, each 2pt. Can directly use the score from Github Classroom

Performance grading (32pt)
--------------------------
* Following equation in assignment PDF
