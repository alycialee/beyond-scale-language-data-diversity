# A Systematic Study of the Role of Data Quality, Alignment and Diveristy for Fine-tuning LLMs for Enhanced Autoformalization

Brando Miranda: brando9@stanford.edu
AI/ML
Aut_win_spr, 2023-2024 Academic Year
Course credit
Up to 5 students

# Project Description
This project embarks on a systematic/scientific exploration of data quality, diversity and data alignment to optimize the performance of Large Language Models (LLMs) in Autoformalization, the process of translating informal natural language statements into formal, verifiable statements (e.g., Python, Lean, Coq, Isabelle). 
The endeavor aims to systematically identify the type of data that maximizes test performance for Autoformalization, utilizing Task2Vec data alignment coefficients to measure alignment between source and target tasks. 
The hypothesis that the most aligned data yields the most significant improvement will be rigorously tested, exploring varying degrees and types of data, including exactly aligned data, unpaired data, formal data alone, and informal data alone, each with different alignment scores. 
The ultimate aspiration is to construct an automated mathematician capable of advancing mathematics, scientific discovery, and AI safety, with a conjecture that formal mathematics is pivotal for creating safe AGI, as it can provide guarantees that are impossible to achieve empirically.
Given the current success of LLMs trained at large scale, we believe a Data Centric Approach to ML and Autoformalization is the paramount.
We conjecture that the alignment coefficient could give a paradigms of data efficiency in machine learning models, paving the way for breakthroughs that transcend existing scaling laws.

# Recommended Background
Candidates are encouraged to share their background when reaching out. 
A robust foundation in Python is crucial, and familiarity with theorem proving using Lean, Coq, or Isabelle is advantageous but not obligatory. 
A fervent interest or profound curiosity in mathematics, formalization/verification of mathematics, AI safety/alignment, or software verification & verified program synthesis is highly desirable.
A passion for Data Centric Mahcine Learning is a plus.

Key citations:
1. Autoformalization (AF): https://arxiv.org/abs/2205.12615
2. AF video: https://youtu.be/_pqJYnQua58?si=jVliUTqqXTjpeods&t=1
3. Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data: https://arxiv.org/abs/2306.13840
4. The Curse of Low Task Diversity: On the Failure of Transfer Learning to Outperform MAML and Their Empirical Equivalence: https://slideslive.com/38996684/the-curse-of-low-task-diversity-on-the-failure-of-transfer-learning-to-outperform-maml-and-their-empirical-equivalence?ref=search-presentations-low+diversity
5. Task2Vec: Task Embedding for Meta-Learning https://arxiv.org/abs/1902.03545

# Prerequisites / Preparation
Participants should be adept at coding in Python and are expected to make substantial contributions to the project. 

# Experiments

## Plan/Experiment 1: Static eval for AutoFormalization (AF) using NLP equivalence score/loss
Goal: first plan will be to use the AF data in [ProoNet](https://huggingface.co/datasets/hoskinson-center/proofnet) or my [debug1](https://huggingface.co/datasets/brando/debug1_af) to evaluate a models capabilities in Autoformalizing using a standard NLP loss function as the equivalence function. 

See dummy code here: https://github.com/brando90/evals-for-autoformalization/blob/main/src/nlp_eval/af_ppl_eval.py

```python
af_score = eval_af_static(model, equi_score_or_loss, eval_dataset, env=LeanDojo)
print(f'Autoformalization eval performance: {af_score=}')
```

TODO, improve bellow:
We will also test it in terms of conceptual categories to see which improve AF most e.g.,
- fine-tune on exactly algined data i.e., on x=(informal_smtmt, formal_stmt) pairs
- fine-tune on seperate unpaired data i.e., trained on both informal and formal statmenets but the data isn't paired (i.e. don't correspond to each other)
- on formal data alone of different alignment scores e.g., python code, pytoch + docs, lean, lean + docs, Coq, cvc4, z3, fol, Isabelle, metamath, mizar, proof terms, proof states, etc.
- on informal data alone, irrelevant data to mathematics, relevant but informal, mathematics textbooks 
- later use equivalence & proof acc to test our method
- maybe also the role of data diversity with the Task2Vec diversity coefficient
- fine-tuning done on lamma2 or Falcon, 1B or 7B, 13B probably max, ping me for code
- note: all of the above should be done with in context learning/few shot condition besides fine-tuning i.e., select the seqs/batch to include in the prompt for AF
- pre-training or continued pre-training can be done too when the focus is testing the diversity coefficient
- AF diversity can be tested by having many AF challenges divided by textbooks or concepts

## Appendix
code refs:
- https://github.com/brando90/beyond-scale-language-data-diversity/blob/main/src/alignment/align.py
