---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

# LM-class

Version: 2024.1beta

LM-class is an education resource for contemporary language modeling, broadly construed. The materials were developed for Cornell CS 5740 Natural Language Processing, and were first taught in Spring 2024. This class constituted a significant re-formulation of CS 5740. The target audience of the class is MEng/MS graduate students. The material is likely also appropriate for senior undergraduates. While the technical material is a good fit for PhDs and researchers, this class does not focus on research, and is designed primarily for practitioners.

The class emphasizes technical depth rather than coverage. It does not aim to provide a broad overview of everything that is happening in the field. The objective is to give students a strong base and the tools to expand their knowledge and update it on their own.

## Lectures

The [lectures](lectures) are organized into three sections:

1. Warming up: this section quickly brings the students up to speed with the basics. The goal is to prepare the students for the first assignment. Beyond a quick introduction, it includes: data basics, linear perceptron, and multi-layer perceptron.
2. Learning from raw data: this section focuses on representation learning from raw data (i.e., without any annotation or user labor). It is divided into three major parts: word embeddings, next-word-prediction language modeling, and masked language modeling. Through these sub-section we introduce many of the fundamental technical concepts and methods of the field.
3. Learning from annotated data: this section focuses on learning with annotated data. It introduces the notion of task as a framework to structure solution development, through the review of several prototypical NLP tasks. This section takes a broad view of annotated data, including covering language model alignment using annotated data (i.e., instruction tuning and RLHF).

## Assignments

The class includes four assignments. The assignments are available upon request. The assignments were developed by [Anne Wu](https://annshin.github.io/) and [Omer Gul](https://momergul.github.io/).

Each assignment includes instructions, report template, starter repository, code to manage a leader board, and a grading schema. The assignments emphasize building, and provide only basic starter code. Each assignment is designed to compare and contrast different approaches. The assignments are complementary to the class material, and require significant amount of self-learning and exploration.

The leader board for each assignment uses a held-out test set. The students get un-labeled test examples, and submit the labels for evaluation on a pre-determined schedule. We use Github Classroom to manage the assignment. Submitting results to the leader board require committing a results file. Because we have access to all repositories, computing the leader board simply require pulling all students repositories at pre-determined times. Github Classroom also allows us to include unit tests that are executed each time the students push their code. Passing the unit tests is a component of the grading.

The assignments include significant implementation work and experimental development. We mandate a minimal milestone 7-10 days before the assignment deadline. We observed that this is a necessary forcing function to get students to work on the assignment early.

Grading varies between assignment. Generally though, it is made of three components: automatic unit tests (i.e., autograding), a written report, and performance on the test set. Performance is a significant component of the grading and follows a per-assignment formula. We generally do not consider leader board placement for grading, except a tiny bonus for the top performer.

### Assignment 1

This assignment is released promptly at the beginning of the semester. The students are asked to implement, experiment with, and contrast a linear perceptron and multi-layer perceptron (i.e., a neural network) for text classification tasks. The linear perceptron is implemented from scratch with external framework (i.e., no PyTorch, SciPy, or NumPy). The students learn how the properties of language support efficient implementation of methods, how to design features, how to conduct error analysis, and how manual features compare to learned representations. This assignment currently does not have a milestone, but we plan to add one for the next iteration of the course.

### Assignment 2

The focus of this assignment is language modeling (i.e., next-word prediction). The students implement, experiment with, and contrast an n-gram LM a transformer-based LM. We ask the students to experiment with different word-level n-gram LMs by varying the n-gram size, smoothing techniques, and tokenization (words vs. sub-words). We also ask them to contrast n-gram and transformer-based LMs. Because of the computational requirements of neural models, this comparison is done on character-based LMs. We provided a buggy implementation of the transformer block. The students were asked to fix it, and build the LM on top of it. We provided autograding unit tests to evaluate progress on fixing the transformer block. We provided two benchmarks to evaluate on: perplexity on a held-out set and word error rate (WER) on a speech recognition re-ranking task. The leader board used the recognition task, because perplexity is sensitive to tokenization.

### Assignment 3

The focus of this assignment are self-supervised representations. The students compare word2vec, BERT, GPT-2 representations. They implement and train word2vec, and experimentally compare it to off-the-shelf BERT and GPT-2 models. The goal of implementing word2vec is to develop a deep technical understanding of estimating self-supervised representations and observe the impact of training design decisions, within the resources available to students. The goal of comparing the three models is to gain insights into the quality of their representations. The experiments use context-independent and context-dependent word similarity datasets, exposing the impact of the different training paradigms.

### Assignment 4

The focus of this assignment is supervised sequence prediction. It uses a language-to-SQL code generation task using the ATIS benchmark, with evaluation of both surface-form correctness and database execution. The student implements three approaches to the problem: prompting with an LLM (including ICL), fine-turning a pretrained encoder-decoder model, and training the same model from scratch. The assignment uses a T5 model that is provided to the students, and training from scratch uses the same model but with randomly initialized parameters.

## Planned Additions

There are several topics not covered by the current version of the lectures, and that are on the planning board to potentially be added in the future:

1. Mixture of experts (MoE)
2. Approximate fine tuning and inference (e.g., LoRA, QLoRA)
3. State space models (SSMs)
4. RNNs and CNNs

I am also considering adding a PyTorch tutorial, with focus on:

1. Setting up experiments
2. Handling data
3. Conducting parameter sweeps
4. Basic debugging techniques
