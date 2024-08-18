Assignment 2 Rubric
===================

[3pt] Section 1: Introduction
-----------------------------
* -0.4pt per
  * Missing the LM task (evaluated with perplexity)
  * Missing the ASR transcription reranking task (evaluated with WER)
  * Does not specify LM task evaluated on WSJ data
  * Does not specify reranking task evaluated on HUB data (can revisit)
* -1.5 per
  * No paragraph summarizing data and tasks
  * No paragraph summarizing main experiments and results

[11pt] Section 2: Data
----------------------

### [3pt] Data
* -0.3 per
  * Missing number of examples for one split (the “missing stats” refers to the number of examples here)
    * WSJ: train/dev/test
  * Not reporting word-level statistics
    * In particular: check that the students didn’t use character-level sentence length (a sentence length of ~120+ is probably char-level)
* -0.25 per (only WSJ)
  * Not specifying the vocabulary size
  * Not specifying the sentence/doc length
  * Not specifying how they computed a statistics that can be computed in different ways (vocabulary size)
    * Note: it’s ok if they at least said something like “number of unique lowercase words” or similar. It’s not enough if it’s just “word-level”, because we could have different results depending on if it’s unique/casing

### [8pt] Preprocessing, Tokenization, Handling unknowns
* -1 per
  * No description of preprocessing (casing)
  * No description of preprocessing (tokenization)
  * No statistics after preprocessing (e.g. sentence length, vocabulary size)
* -0.5 per 
  * If used a subset of the data, no explanation of why
  * If used a subset of the data, no discussion of the tradeoffs
  * If used a subset of the data, the explanations are not backed by experiments
  * If they did not mention thresholding for unknown words  
  * If the exact tokenizer they used is ambiguous 
* -1.5 per
  * No discussion of how unknown words are handled
* -1 per
  * Clearly wrong/invalid ways of handling of unknown words
  * No discussion of the trade-offs in handling the unknown words

[12pt] Section 3: Models
------------------------
* -1.5pt per 
  * If the student makes no reference to the smoothing techniques they tried
  * If the student makes no reference to the tokenization strategies they tried for the subword-level model.
* -1pt per
  * If the n-gram objective is not described (maximizing corpus probability, cf. slide 24)
  * If the n-gram MLE solution is not described (cf. slide 28)
  * If the transformer function is not described (they can abstract away the function, but we at least expect the prediction head in notation form)
  * If the transformer objective is not described (the cross entropy loss)
* -0.5pt per
  * If the n-gram objective is partially incorrect or verbally described
  * If the n-gram MLE description is partially incorrect or verbally described
  * If the Transformer function is incorrectly or verbally described
  * If the Transformer’s objective is incorrectly or verbally described (the cross entropy loss)
* -1pt per:
  * If the notation in the section is poor enough to make the content difficult to understand

[4pt] Section 4: Implementation Details
---------------------------------------
* -1pt per:
  * If they did not describe what the tunable hyperparameters for the character-level model were (smoothing techniques; n-gram values)
  * If they did not describe what the tunable hyperparameters for the word-level model were (smoothing techniques; n-gram values; tokenization)
  * If they did not describe what the tunable hyperparameters for the transformer were. At least 2 of: number of layers, hidden dimension, number of heads, learning rate
  * If these were mentioned in the following section, no need to cut points.

[8pt] Section 5: Experiments
----------------------------
* -1pt per (treat character- and word-level n-grams independently):
  * No experiments performed for n=1,2 and 3
  * No experiments performed comparing word-level to BPE tokenization (word-level only)
* -0.5 once
  * If the student does not make comparisons between smoothing and no smoothing
  * If the student did not use linear interpolation or backoff for smoothing experiments
* -0.25 once
  * If the student did not ever use linear interpolation, but used backoff instead.
* -1.5pt
  * If students varied less than 2 hyperparameters for transformers. The set of hyperparameters includes but is not limited to: number of layers, hidden dimension, number of heads, learning rate
* -1pt per (clarity issue):
  * If we cannot reproduce their exact experiments for character- and word-level n-grams (i.e. they didn’t specify the hyperparameter settings)
  * If we cannot reproduce their exact experiments for transformers

[20pt] Section 6: Results and Analysis
--------------------------------------
* -2 per:
  * Missing entry in dev for the best model (per system - character-level n-gram, word-level n-gram, transformer)
  * Missing entry in dev for a model variant specified
  * Missing test results or reports non-final results (compared to leaderboard) on test set
  * If the students were penalized for missing an entry in dev for the best model, do not double penalize for missing an entry in dev for a variant.
* -1 per system (character-level n-gram, word-level n-gram, transformer):
  * Unclear/invalid model variant
* -1 per not discussing or invalid discussion of contrasting choice for: 
  * WSJ
    * N-gram
      * Smoothing
        * Within-system comparison for transformer (can choose not to cut points if the trends are clear from the table)
  * HUB
    * N-gram
    * Tokenization
    * Smoothing
    * Neural vs. n-gram
* -1 once
  * If discussion results don’t match what’s on the table
* -3 per
  * System missing entirely from discussion (would not be applying -1per above for that system) (can revisit the score)

[3pt] Section 7: Conclusion
---------------------------
* -3 if no conclusion
* -1 pt:
  * No summative statement comparing overall systems

[10pt] Autograding
------------------
* Directly taking the points from Github Classroom (5 unittests, 2pt/unittest)

[24pt] Performance
------------------
* Following equation in assignment PDF


[5pt] Performance milestone
---------------------------
* -5pt if:
  * The performance on the leaderboard at 12:00am Mar 11 (deadline was: 11:59pm Mar 10) is > 13.3% WER
