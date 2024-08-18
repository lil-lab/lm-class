Assignment 3 Rubric
===================

Section 1: word2vec training (16pt)
-----------------------------------
* Applicable for both Table 1 and 2
  * Applicable to every row:
    * -1 per
        * No or invalid entry / description
  * For “Context construction”
    * -0.5 per
      * No description of how the context is constructed
      * Not specifying the context window size
      * Embedding size not following the instruction (> 1024)
  * For “Negative sampling”
    * -0.5 per
      * Not specifying the distribution from which the negative samples are sampled
      * Not specifying how many negative examples are used per context-word pair
  * For “Dataset construction”
    * -0.3 per
      * Not specifying which dataset (i.e. 1M sentences or the extended) is used
      * If used the extended dataset: not specifying how many sentences were sampled or how
      * Not specifying the final dataset’s size (for the 1M case, it’s ok if the student just said that they’re using the base dataset with 1M sentences)
  * For “Data preprocessing”
    * -0.3 per
      * Not specifying how casing is handled
      * Not specifying how tokenization is handled
      * Not specifying how unknown words are handled
      * Unknown word handling doesn’t make sense
  * For “Training hyperparameters”
    * -0.5 per
      * No or invalid description
    * -0.25 per
      * Partially invalid or wrong description
      * Stopping criterion being just a predefined number of epochs and no explanation

Section 2: representation extraction from BERT (10pt)
-----------------------------------------------------
* Applicable for both Table 3 and 4
  * Applicable to every row (unless otherwise specified in section-wise rubric below)
    * -1 per
      * No or invalid entry / description
    * -0.5 per
      * Partially invalid or vague description (can I replicate this in PyTorch based on the description?)
  * For “Combining embeddings from multiple layers”
    * Note: if they didn’t do combination and thus didn’t put a description, we don’t penalize

Section 3: representation extraction from GPT-2 (10pt)
------------------------------------------------------
Same rubric as Section 2 for Table 5 and 6

Section 4: results for isolated word pairs (7pt)
------------------------------------------------
* Result table
  * -1 per
    * Test results don’t match the leaderboard (either result, capped at -1)
    * Test results shown instead of dev results in the dev result section
    * Not reporting the right results for one model (e.g. exactly the same test and dev results for all the models, capped at -3)
  * -0.5
    * Description of variant vague or hard to understand (it’s ok if they specified in the caption)
* Word2vec:
  * Number of variants (at least 3 experiments from the provided list are required, i.e. training data size, embedding dimension, context window size, number of negative sample)
    * -1 if experimented with only one of the choices
    * -0.5 if experimented with only two of the choices
    * -1.5 if not experimenting with required choices
  * Only for isolated word pairs (Table 7)
    * Test score much better than dev score (e.g. ~3x than dev score, with test score >0.3)
* Applicable to BERT/GPT-2 (per model)
  * Number of variants: They must experiment with four choices: what layer(s) to extract embeddings from, how to combine subword embeddings, whether to do cosine similarity (ie. unit normalization or not) and how to combine multiple layers if they use multiple layers.
    * -0.25 per
      * If there are no experiments comparing normalization to no normalization
      * If there are no experiments exploring different strategies for merging subwords
      * If there are no experiments varying what layer to use
      * If their full system does not use more than one layer, penalize if they don’t have an experiment comparing performance against multiple layers
      * If their full system does use more than one layer, penalize if they don’t have experiments comparing different strategies of merging layers
* Plot
  * -1 for no plot
  * -0.3 per
    * No figure caption
    * Missing or vague axis labels (e.g. when the x-axis is “step”, but no specification of whether one step is an epoch, a minibatch or else)

Section 5: results for contextualized word pairs (7pt)
------------------------------------------------------
Same rubric as Section 4

Milestone (6pt)
---------------
Will lose points if milestone score below 0.1

Performance grading (44pt)
--------------------------
* Following equation in assignment PDF
