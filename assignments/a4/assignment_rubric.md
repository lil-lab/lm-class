Assignment 4 Rubric
===================

Section 1: word2vec training (2pt)
-----------------------------------
* -2 pt if missing (they can copy results from A2)
* deduct up to 2 if there are other errors. 
  

Section 2: representation extraction from BERT (12pt)
-----------------------------------------------------
* Applicable for both Table 3 and 4
  * Applicable to every row (unless otherwise specified in section-wise rubric below)
    * -1 per
      * No or invalid entry / description
    * -0.5 per
      * Partially invalid or vague description (can I replicate this in PyTorch based on the description?)
  * For “Combining embeddings from multiple layers”
    * Note: if they didn’t do combination and thus didn’t put a description, we don’t penalize

Section 3: representation extraction from GPT-2 (12pt)
------------------------------------------------------
Same rubric as Section 2 for Table 5 and 6

Section 4: results for isolated word pairs (5pt)
------------------------------------------------
* Result table
  * -1 per
    * Test results don’t match the leaderboard (either result, capped at -1)
    * Test results shown instead of dev results in the dev result section
    * Not reporting the right results for one model (e.g. exactly the same test and dev results for all the models, capped at -3)
  * -0.5
    * Description of variant vague or hard to understand (it’s ok if they specified in the caption)
* Applicable to BERT/GPT-2 (per model)
  * Number of variants: They must experiment with four choices: what layer(s) to extract embeddings from, how to combine subword embeddings, whether to do cosine similarity (ie. unit normalization or not) and how to combine multiple layers if they use multiple layers.
    * -0.5 per
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

Section 5: results for contextualized word pairs (8pt)
------------------------------------------------------
Same rubric as Section 4, except that the plot is optional. 

Milestone (6pt)
---------------
Will lose points if milestone score below 0.15. Only consider contextual embeddings from GPT2 or BERT. 

Performance grading (45pt)
--------------------------
* Following equation in assignment PDF
