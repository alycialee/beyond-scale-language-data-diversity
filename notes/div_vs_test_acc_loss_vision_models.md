# How does the Diversity training set affect the downstream performance on a model -- in depth study by comparing trained checkpoints on a fix evaluation set

# Recommended Background

Candidates are encouraged to share their background when reaching out. A strong foundation in Python is essential, and a belief in the paramount role of data in Machine Learning, aligning with data-centric ML principles, is a plus.

Key citations:
- Is pre-training truly better than meta-learning? https://arxiv.org/abs/2306.13841 
- Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data: https://arxiv.org/abs/2306.13840
- The Curse of Low Task Diversity: On the Failure of Transfer Learning to Outperform MAML and Their Empirical Equivalence: https://slideslive.com/38996684/the-curse-of-low-task-diversity-on-the-failure-of-transfer-learning-to-outperform-maml-and-their-empirical-equivalence?ref=search-presentations-low+diversity
- Task2Vec: Task Embedding for Meta-Learning https://arxiv.org/abs/1902.03545
- Beyond neural scaling laws: beating power law scaling via data pruning: https://arxiv.org/abs/2206.14486

# 1 High level experiment description
Hypothesis/Goal: the hypothesis is that data is central to ML, in particular a diverse/general data set is required to achieve genereral/diverse intelligence (~AGI).
Therefore, we test hypothsis this concretely with model that we know their train diversity.
For this we require all models train in a similar manner (e.g., same architecture/params, same opt, same convergence criteria, same tokens, or as many controlled variables) so that we know the source of variance in test performance is most likely the diversity.
We also need to choose an eval set that does not intersect with the train set -- or if it does intersect, that we understand how or are able to justify it such that our experiments still are meaningful with respect to the central hypothesis (role of formal div in test acc).
For example, if the eval data set has cifar, but all models compared were trained on cifar in the same way, then the comparison is fair. What is not fair is to have a model trained on meta-data set (MDS), test on meta-data set (MDS) but then the other models know nothing about MDS.
So the MDS model likely will perform best because it was trained on the test/eval set, not because it's training diversity is high. 

## 1.1 Train Diversity Coeff  vs Test Accuracy/loss -- using Pre-traing vision Checkpoints
Goal: The goal is to produce a single plot where on the x-axis we have the diversity coefficient of the training data set used to produce the model or checkpoint vs in the y-axis the test/val accuracy/CE-loss on a diverse evaluation/test/val set.
Thus, the goal is to see how diversity of the training data affect downstream performance for a general/diverse eval/test set (~ as a proxy for "general intellgience"/AGI).

Experiment set up suggestion:
- x-axis model trained with diversity (but they are all the same model e.g., all resnet12's)
- y-axis all models evaluated on the same eval/test set with the same metric (suggest CE-loss and few-shot accuracy)
- constraints/controlls for fair comparisons:
  - same architecture/params, same opt, same convergence criteria, same tokens, or as many controlled variables
  - intersection/overlap of eval and train is the same (ideally test has no intersection, but if the error/bias is systematic experiments are meaningful)
 
Note: checkpoints were NOT controlled for data points, but experiments are still meaningful because:
- TODO: think of argument
- But, if you want you **can** re-train models (I suggest pre-trained (PT) models, not MAML models). But data points might not be the best way to control, maybe a sense of "complexity" or "information".
- But it's better to have some results than none-since controlling for "info" is hard anyway while changing diversity at the same time.

### Models
Models can be found in my Zenodo:
- https://zenodo.org/records/8247898 
Models can be found in my Hugging Face (HF):
- https://huggingface.co/brando
gdrive:
- https://drive.google.com/drive/folders/15cuk9Zu455DXNMVBMmRpZdUZUCSImsmf
I suggest you download these to the servers you will use programatically i.e., with Python/Bash. It's easy if you ask GPT4/Claude.

## 1.2 Train Diversity Coeff  vs Test Accuracy/loss -- using MAML vision Checkpoints
Same as in 1.1 but using MAML checkpoints
