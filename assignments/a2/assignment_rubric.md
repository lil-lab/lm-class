Assignment 3 Rubric
===================

Section 1: word2vec training (20 pt)
-----------------------------------
* Applicable for both Table 1 and 2
  * For Row 1 and Row 7: 
    * -2.5 per
      * No or invalid entry / description
  * For Row 2-6:
    * -3 per
        * No or invalid entry / description
  * For “Context construction”
    * -1 per
      * No description of how the context is constructed
      * Not specifying the context window size
  * For “Negative sampling”
    * -1 per
      * Not specifying the distribution from which the negative samples are sampled
      * Not specifying how many negative examples are used per context-word pair
  * For “Dataset construction”
    * -1per
      * Not specifying which dataset (i.e. 1M sentences or the extended) is used
      * If used the extended dataset: not specifying how many sentences were sampled or how
      * Not specifying the final dataset’s size (for the 1M case, it’s ok if the student just said that they’re using the base dataset with 1M sentences)
  * For “Data preprocessing”
    * -1 per (up to -3)
      * Not specifying how casing is handled
      * Not specifying how tokenization is handled
      * Not specifying how unknown words are handled
      * Unknown word handling doesn’t make sense
  * For “Training hyperparameters”
    * -3 per
      * No or invalid description
    * -1 per
      * Partially invalid or wrong description
      * Stopping criterion being just a predefined number of epochs and no explanation



Section 2: Results (20 pt)
------------------------------------------------
* Result table
  * -1 per
    * Test result on the leaderboard is lower than the test score on the report (don’t deduct points if the leaderboard score is higher.)
    * Test result shown instead of dev result in the dev result section
    * Not reporting the right results (e.g. exactly the same test and dev results)
  * -1
    * Description of variant vague or hard to understand (it’s ok if they specified in the caption)
* Word2vec:
  * Number of variants (at least 3 experiments from the provided list are required, i.e. training data size, embedding dimension, context window size, number of negative sample)
    * -10 if experimented with only one of the choices
    * -5 if experimented with only two of the choices
    * -15 if not experimenting with required choices
  * Only for isolated word pairs (Table 7)
    * -3 if final model test score much better than dev score (e.g. ~3x than dev score, with test score >0.3)
* Plot
  * -5 for no plot
  * -1 per
    * No figure caption
    * Missing or vague axis labels (e.g. when the x-axis is “step”, but no specification of whether one step is an epoch, a minibatch or else)



Milestone (6pt)
---------------
* -3 if between 0.08 and 0
* -6 if below 0

Performance grading (44pt)
--------------------------
* Following equation in assignment PDF
* Can be more than 44pt (allow bonus points)



### Late Penalty

* TBD

