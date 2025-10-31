---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: Assignments
order: 20
---

The class includes five assignments. The assignments are all available on the LM-class repository: [lm-class/tree/main/assignments](https://github.com/lil-lab/lm-class/tree/main/assignments). The repository does not include the test data. Please email [Yoav Artzi](mailto:yoav@cs.cornell.edu) for the test data. It is critical for the function of the class that the test data remains hidden.

Each assignment includes instructions, report template, starter repository, code to manage a leader board, and a grading schema. The assignments emphasize building, and provide only basic starter code. Each assignment is designed to compare and contrast different approaches. The assignments are complementary to the class material, and require significant amount of self-learning and exploration.

The leader board for each assignment uses a held-out test set. The students get unlabeled test examples, and submit the labels for evaluation on a pre-determined schedule. We use GitHub Classroom to manage the assignment. Submitting results to the leader board requires committing a results file. Because we have access to all repositories, computing the leader board simply requires pulling all students repositories at pre-determined times. GitHub Classroom also allows us to include unit tests that are executed each time the students push their code. Passing the unit tests is a component of the grading.

The assignments include significant implementation work and experimental development. We mandate a minimal milestone 7-10 days before the assignment deadline. We observed that this is a necessary forcing function to get students to work on the assignment early.

Grading varies between assignments. Generally though, it is made of three components: automatic unit tests (i.e., auto-grading), a written report, and performance on the held-out test set. Performance is a significant component of the grading and follows a per-assignment formula. We generally do not consider leader board placement for grading.

## Assignment 1

This assignment is released promptly at the beginning of the semester. The students are asked to implement, experiment with, and contrast a linear perceptron and multi-layer perceptron (i.e., a neural network) for text classification tasks. The linear perceptron is implemented from scratch without an external framework (i.e., no PyTorch, SciPy, or NumPy). The students learn how the properties of language support efficient implementation of methods, how to design features, how to conduct error analysis, and how manual features compare to learned representations.

## Assignment 2

The focus of this assign is context-independent self-supervised representations. The students implement and train word2vec. The goal of implementing word2vec is to develop a deep technical understanding of estimating self-supervised representations and observe the impact of training design decisions, within the resources available to students. This gives the students a tractable taste of a large-scale ML experiment, and forces them to think of extracting the best from their data given available resources. 

## Assignment 3

The focus of this assignment is language modeling (i.e., next-word prediction). The students implement, experiment with, and contrast an n-gram LM and a transformer-based LM. We ask the students to experiment with different word-level n-gram LMs by varying the n-gram size, smoothing techniques, and tokenization (words vs. sub-words). We also ask them to contrast n-gram and transformer-based LMs. Because of the computational requirements of neural models, this comparison is done on character-based LMs. We provide a buggy implementation of the transformer block. The students are asked to fix it, and build the LM on top of it. We provid auto-grading unit tests to evaluate progress on fixing the transformer block. There are two modes of evaluation: perplexity on a held-out set and word error rate on a speech recognition re-ranking task. The leader board uses the recognition task, because perplexity is sensitive to tokenization.

## Assignment 4

This assignment re-visits token-level representations, including the representations induced in Assignment 2. The students compare word2vec, BERT, GPT-2 representations. They use the word2vec representations from Assignment 2. The goal of comparing the three models is to gain insights into the quality of their representations, and to the fundamental contrast between context-dependent and -independent representations. In addition to the context-independent similarity benchmarks from Assignment 2, we add context-dependent word similarity, exposing the impact of the different training paradigms.

## Assignment 5

This assignment focused on supervised sequence prediction. It uses a language-to-SQL code generation task using the ATIS benchmark, with evaluation of both surface-form correctness and database execution. The student implements three approaches: prompting with an LLM (including ICL), fine-tuning a pretrained encoder-decoder model, and training the same model from scratch. The assignment uses a small T5 model that is provided to the students, and training from scratch uses the same model but with randomly initialized parameters.
