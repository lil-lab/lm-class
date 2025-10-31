Assignment 4 Rubric
===================

Section 1 (8pt total)
---------------------
* -0.5pt per 
  * Missing row on Table 1
  * Missing row on Table 2 for T5-FT
    * Do not need to report number of examples
  * Missing row on Table 2 for T5-From Scratch
  * Missing row on Table 2 for T5-FT
* -0.25pt per
  * Invalid row on Table 1
  * Invalid row on Table 2 for T5-FT
  * Invalid row on Table 2 for T5-From Scratch
  * What is an invalid entry for Table 1?
    * For SQL query length: If they use character-level tokenization without specifying it (> 500 tokens)
    * For NL query length: If they use character-level tokenization without specifying it (> 30 tokens)
    * Vocabulary size (both): If they use the total model vocabulary size (~31k)
  * What is an invalid entry for Table 2?
    * For SQL and NL query length: Using word or character-level statistics
    * Vocabulary size (both): If they use the total model vocabulary size (~31k)
* -1pt once
  * They provide the T5 lexicon as the vocabulary size for everything
* -2.5pt per
  * Missing table (Table 1, Table 2 for T5-FT and Table 2 for T5-From Scratch)
    * It is ok if they merge the T5-FT and T5-From Scratch reporting

Section 2 (8pt total)
---------------------
* -4pt per
  * Empty table
* -1pt per
  * Missing entry in a row (blank or original instruction text kept in full)
    * It is ok if they say “Same as above” etc
    * It is also ok if they say “No data preprocessing,” etc. We mainly want them to report what they did in case they went beyond just using the T5Tokenizer/doing full finetuning.
* -0.5pt per table
  * Addition of a BOS token not mentioned either in data processing or tokenization
  * At least the following are not described: learning rate, stopping criterion and batch size
  * Stopping criterion a fixed number of epochs without reference to model selection

Section 3 (14pt total)
----------------------
### 3.1 ICL: Table 5 (3pt)
* It’s ok if:
  * they just put the k-shot prompt, as long as it’s clear that their k-shot and zero-shot prompts are constructed in the same way, and the prompt is clearly understandable
  * they either provide concrete examples or placeholder tokens, as long as it’s clear what the tokens correspond to
* -3pt
  * Empty table
* -0.25
  * Not specifying the value of k in Table 5 for a given prompt
* -0.3 per
  * Unclear how to parse the prompt
  * Incomplete prompt (e.g. just providing part of the prompt)
* -1 per
  * Not providing the k-shot prompt

### 3.1 ICL: Example selections (3pt)
* -3 per
  * No description of how the examples are selected
* -1
  * Invalid description of example selections
  * Not selecting examples from the training set
* -0.5
  * Vague description of example selections

### 3.2 Best Prompt: Table 6 (3pt)
* -3 per
  * Empty table
* -0.5 per
  * Unclear specification of the prompt (e.g. the prompt is not exactly the same as the one in Table 5, but it’s hard to understand the differences, or if the description is vague)

### 3.2 Ablation Study: Table 7 (5pt)
* -5
  * Empty table
* -0.3 per
  * If the ablation has not been highlighted in Table 6
  * If they indicated in a clear way what are the ablation variants (e.g. use a table to distinguish between the different prompts, that’s fine)
* -0.2
  * < 2 ablation experiments
  * Unclear or vague description of an ablation

Section 4 (20pt total)
----------------------

### 4.1, Table (10pt total)
* -2pt per system:
  * No dev set F1 result reported at all
* -1pt per system:
  * Test result does not match what is on the leaderboard
  * No test result given
* -1pt once
  * Query EM Score missing
  * Query EM reported to be higher than F1 (make sure you know what the columns represent)
* -0.5pt once
  * Description of row or variant hard to understand
* T5-Finetuning
  * -1pt per
    * If they report selectively finetuning certain layers and do not have ablation experiments corresponding to this parameter
    * If they report performing data processing/tokenization beyond just applying T5Tokenizer and do not have ablation experiments supporting their choice
* T5 From Scratch
  * -1pt per
    * If they report performing data processing/tokenization beyond just applying T5Tokenizer and do not have ablation experiments supporting their choice

### 4.2, Plot (4pt total)
* -4pt once
  * No plot
* -2pt once
  * If they provide only one datapoint for k
* -1pt once
  * If they provide only two datapoints for k
  * It is very difficult to understand the plot (cannot understand what axes represent, etc)
* -0.5pt
  * Not at least trying the values of k that are asked for (k=0, 1, 3)

### 4.3, Error analysis (6pt total)
* -2pt per
  * Missing row (must have at least 3)
  * Missing system (must report something for at least each system)
* -1.5pt once
  * No statistics provided
  * Only provided examples drawn from SQL error messages (and therefore did not actually do analysis)
  * SQL queries never shown in the “Example of Error” section
* -1pt once
  * Error description too unclear
  * Statistics provided but invalid (rough estimates/no specification of sample size)
  * If multiple systems are given in one row, statistics provided for one of them
* -1pt per
  * Invalid error category

Performance (45pt)
------------------
* Following equation in assignment PDF

Milestone (5pt)
---------------
* Following requirement in assignment PDF