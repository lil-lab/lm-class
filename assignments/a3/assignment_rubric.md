Assignment 3 Rubric
===================

Total: 100 points

Q1 Data (5 pt)
--------------
* -1 Missing vocab size
* -1 Missing sentence/document length
* -1 Missing # of examples
* -1 Missing or nonsensical comment on how any statistic was computed
* -1 Missing ASR data



Q2 N-gram Models (14 pt)
------------------------
* Number of model variants:
  * -3 Between 5 and 7 (inclusive) model variants
  * -6 Between 2 and 4 (inclusive) model variants
  * -12 Only one model
  * -14 No model variants
* -5 Missing both smoothing-less experiment and linear-interpolation experiment (Kneser-Ney counts as linear interpolation)
* Value of n:
  * -8 Only experimented with one value of n
  * -4 Only experimented with two values of n
* -7 Only experimented with one form of tokenization
* Smoothing:
  * -1 Missing experiments without smoothing
  * -2 Missing experiments without smoothing (only Linear Interpolation)
  * -2 Lack of an accurate description of the smoothing technique
  * -2 Missing linear interpolation smoothing



Q3 Transformer LMs (10 pt)
--------------------------
* Number of model variants:
  * -3 Only 4 or 5 (inclusive) model variants
  * -6 Only 2 or 3 model variants
  * -8 Only one model variant
  * -10 No model variants
* -2 Nonsensical model description (e.g., impossible hidden dimension, etc.)



Q4 Results (20 pt)
------------------
* -2 Test result on the leaderboard is lower than the test score on the report (don't deduct points if the leaderboard score is higher)
* -5 Final model test score is much better than dev score (e.g., 1/2x dev score)
* Number of variants:
  * -2 Between 10 and 13 (inclusive) variants
  * -3 Between 6 and 9 (inclusive) variants
  * -5 Between 1 and 5 (inclusive) variants
  * -20 No model variants
* Test results coverage:
  * -1 per incomplete row (test results only needed for three total model variants)
  * -5 Test results only reported for 2 models
  * -7 Test results only reported for one model



Q5 Milestone (5 pt)
--------------------
* -3 Score above 13.3
* -5 Score above 15.0



Q6 Autograder (10 pt)
---------------------
* Missing unit tests:
  * -2 Missing one unit test
  * -4 Missing two unit tests
  * -6 Missing three unit tests
  * -8 Missing four unit tests
  * -10 Missing five or more unit tests



Q7 Performance Grading (36 pt)
------------------------------
* Correct (score-based grading)



### Late Penalty

* TBD
