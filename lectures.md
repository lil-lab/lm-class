---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: Lectures
order: 10
---

The lectures constitute a coherent sequence, where later sections often assume concepts and material from earlier sections. They are organized into three sections:

1. Warming up: this section quickly brings the students up to speed with the basics. The goal is to prepare the students for the first assignment. Beyond a quick introduction, it includes: data basics, linear perceptron, and multi-layer perceptron.
1. Learning with raw data: this section focuses on representation learning from raw data (i.e., without any annotation or user labor). It is divided into three major parts: word embeddings, next-word-prediction language modeling, and masked language modeling. Through these subsections we introduce many of the fundamental technical concepts and methods of the field.
1. Learning with annotated data: this section focuses on learning with annotated data. It introduces the task as a framework to structure solution development, through the review of several prototypical NLP tasks. For each task, we discuss the problem, data, modeling decisions, and formulate a technical approach to address it. This section takes a broad view of annotated data, including covering language model alignment using annotated data (i.e., instruction tuning and RLHF).


I am collecting a [document of issues](https://docs.google.com/document/d/1aAYaRvR1BauC4RS5TzCeM4fCbTbnPwQVcjlMAVMlTjU/edit?usp=sharing), including with feedback from other researchers and instructors. It is recommended to consult this document if using this material. I did not review the issue document in depth, so cannot stand by it. However, I plan to review it and address issues in the next iteration of the class (usually: next spring). 

## Warming Up

This section quickly brings the students up to speed with the basics. The goal is to prepare the students for the first assignment. Beyond a quick introduction, it includes: data basics, linear perceptron, and multi-layer perceptron.

<dl>
<dt><strong>Introduction</strong> <a href="/lectures/01 - intro.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/01 - intro.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>A very brief introduction to the class, including setting up the main challenges with natural language and the history of the field.</dd>
<dt><strong>Text Classification, Data Basics, and Perceptrons</strong> <a href="/lectures/02%20-%20data%20basics%20and%20perceptron.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/02%20-%20data%20basics%20and%20perceptron.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>The basics of text classification and working with data splits. We introduce the linear perceptron, starting from the binary case and generalizing to multi-class.</dd>
<dt><strong>Neural Network Basics</strong> <a href="/lectures/03%20-%20neural%20networks.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/03%20-%20neural%20networks.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>A very quick introduction to the basics of neural networks and defining the multi-layer perceptron.</dd>
</dl>

## Learning with Raw Data

This section focuses on representation learning from raw data (i.e., without any annotation or user labor). It is divided into three major parts: word embeddings, next-word-prediction language modeling, and masked language modeling. Through these subsections we introduce many of the fundamental technical concepts and methods of the field.

<dl>
<dt><strong>Word Embeddings</strong> <a href="/lectures/04%20-%20word%20embeddings.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/04%20-%20word%20embeddings.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>Introduction of lexical semantics. We start with discrete word senses and WordNet, transition to distributional semantics, and then introduce word2vec. We use dependency contexts to briefly introduce syntactic structures.</dd>
<dt><strong>N-gram Language Models</strong> <a href="/lectures/05%20-%20language%20models.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/05%20-%20language%20models.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>We introduce language models through the noisy channel model. We gradually build n-gram language models, discuss evaluation, basic smoothing techniques, and briefly touch on the unknown word problem.</dd>
<dt><strong>Tokenization</strong> <a href="/lectures/06%20-%20tokenization.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/06%20-%20tokenization.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>We discuss handling of unknown words, and from this build to sub-word tokenization. We go over the BPE algorithm in detail.</dd>
<dt><strong>Neural Language Models and Transformers</strong> <a href="/lectures/07%20-%20neural%20lms%20and%20transformers.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/07%20-%20neural%20lms%20and%20transformers.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>This lecture gradually builds neural language models (LM) starting from n-gram models and concluding with the Transformer decoder architecture, which we define in detail. We present attention as a weighted sum of items, previous tokens in this case.</dd>
<dt><strong>Decoding LMs</strong> <a href="/lectures/08%20-%20decoding%20lms.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/08%20-%20decoding%20lms.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>This lecture is a brief discussion of decoding techniques for LMs, mostly focusing on sampling techniques, but also discussing beam search.</dd>
<dt><strong>Scaling up to LLMs</strong> <a href="/lectures/09%20-%20scaling%20up%20to%20llms.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/09%20-%20scaling%20up%20to%20llms.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>We discuss the challenges of scaling LMs to contemporary LLMs, including data challenges, scaling laws, and some of the societal challenges and impacts LLMs bring about. This section focuses on pre-training only.</dd>
<dt><strong>Masked Language Models and BERT</strong> <a href="/lectures/10%20-%20masked%20lms.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/10%20-%20masked%20lms.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>This section introduces BERT and its training. We use this opportunity to introduce the encoder variant of the Transformer.</dd>
<dt><strong>Pretraining Encoder-decoders</strong> <a href="/lectures/11%20-%20encdec%20pretrain.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/11%20-%20encdec%20pretrain.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>This section introduces the BART and T5 models. In the most recent version of the class, it came late in the semester, so it does not define the encoder-decoder Transformer architecture in detail. This content should be migrated from the later tasks slide deck.</dd>
<dt><strong>Working with Raw Data Recap</strong> <a href="/lectures/12%20-%20raw%20data%20recap.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/12%20-%20raw%20data%20recap.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>A very short recap of the first half of the class.</dd>
</dl>

## Learning with Annotated Data

This section focuses on learning with annotated data. It introduces the task as a framework to structure solution development, through the review of several prototypical NLP tasks. For each task, we discuss the problem, data, modeling decisions, and formulate a technical approach to address it. This section takes a broad view of annotated data, including covering language model alignment using annotated data (i.e., instruction tuning and RLHF).

<dl>
<dt><strong>Prototypical NLP Tasks</strong> <a href="/lectures/13%20-%20tasks.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/13%20-%20tasks.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>Defining the task as a conceptual way to think about problems in NLP. We discuss several prototypical tasks: named-entity recognition as a tagging problem, extract question answering as span extraction, machine translation as a language generation problem, and code generation as a structured output generation problem. We use these tasks to introduce general modeling techniques and discuss different evaluation techniques and challenges. We conclude with multi-task benchmark suites. This section currently also defines the encoder-decoder Transformer architecture, as part of the machine translation discussion. This content should be migrated earlier to the encoder-decoder pre-training lecture.</dd>
<dt><strong>Aligning LLMs</strong> <a href="/lectures/14%20-%20aligning%20llms.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/14%20-%20aligning%20llms.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>The process of training LLMs past the initial pre-training stage. We discuss instruction tuning and RLHF. We provide a basic introduction to reinforcement learning, including PPO. We also describe DPO.</dd>
<dt><strong>Working with LLMs: Prompting</strong> <a href="/lectures/15%20-%20prompting.key"><span class="badge text-bg-primary">.key</span></a> <a href="/lectures/15%20-%20prompting.pdf"><span class="badge text-bg-success">.pdf</span></a></dt>
<dd>This lecture covers the most common prompting techniques, including zero-shot, in-context learning, and chain-of-thought.</dd>
</dl>
