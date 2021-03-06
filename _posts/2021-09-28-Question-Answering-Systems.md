---
layout: post
title: "Question Answering Systems"
subtitle: "Who doesnt love a know-it-all AI"
background: '/img/posts/Question-Answering-Systems/background.jpg'
---

## Background

In the Fall of 2022, I had taken the CS528-NLP and as part of this course, we supposed to work on course projects. I will using this post to document my learning on Question Answering Systems.

As part of this project we try explore the diffrent benchmarks as part of [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset.

## Notes from Chapter [23- Question Answering](https://web.stanford.edu/~jurafsky/slp3/)

In following section, lets look into the details given about Question Answering systems from [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) book by Dan Jurafsky.


## Blogs I would like to explore - 

1. [Building a Question-Answering System from Scratch— Part 1](https://towardsdatascience.com/building-a-question-answering-system-part-1-9388aadff507)

She uses sentence embeddings, to find similarity with context & question. She also explores stratergy, parse tree of a sentence is used for getting the answers. And also unsupervised stratergies mentioned.


2. [Building a Question Answering model](https://towardsdatascience.com/nlp-building-a-question-answering-model-ed0529a68c54)

Again uses attention vectors in LSTM & explores BiDAF model

3. [Build a Trivia Bot using T5 Transformer](https://medium.com/analytics-vidhya/build-a-trivia-bot-using-t5-transformer-345ff83205b6)

Explains the 3 diffrent types of QA systems. She talks about 3 stratergies where
    - Compress all the information in the weight parameters. T5 model here
    - For very long context (databases of documents), here we find top-k documents.

4. [Implementing QANet (Question Answering Network) with CNNs and self attentions
](https://towardsdatascience.com/implementing-question-answering-networks-with-cnns-5ae5f08e312b)

Im yet to read this and it probably uses CNN.

## Papers/repo I would like to explore - 

1. [Question Answering on SQuAD](https://github.com/BAJUKA/SQuAD-NLP)
2. [Probably uses huggingace directly for QA](https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/nlp/Question_Answering_Squad.ipynb#scrollTo=n8HZrDmr12_-)
3. [CS 224N Default Final Project: Question Answering](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/default_project/default_project_v2.pdf)
4. [BiDAF Model for Question Answering](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2760988.pdf)

5. [Approach for SQUAD 2.0](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15743593.pdf)
6. [Original SQUAD2.0 paper](https://arxiv.org/pdf/1806.03822.pdf)


## Short Notes on Bidaf Model

[paper](https://arxiv.org/abs/1611.01603) [code](https://github.com/allenai/bi-att-flow)

Paper was introduced in 2016 inorder to solve tasks that query from long context words(Machine Comprehension or Question Answering) and uses attention mechanism to approach the problem.

Important Highlights mentioned in Abstract section - 

> BIDAF includes character-level, word-level, and contextual embeddings,
and uses bi-directional attention flow to obtain a query-aware context representation

> Our experiments show that memory-less attention gives a clear advantage over dynamic attention

> we use attention mechanisms in both directions, query-to-context and context-to-query, which provide complimentary information to each other.

![Model Architecture](/img/posts/Question-Answering-Systems/Model_Architecture.jpg)
### Model

As seen per the figure we pass the model through two seperate backbone networks for each Context and Query sentences. This Backbone in the bottom half uses three seperate encoding mechanism - Character Embedding Layer, Word Embedding Layer & Contextual Embedding Layer

