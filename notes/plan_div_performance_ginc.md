# Beyond Scale II: Evaluating the Impact of Data Quality and Diversity on LLMs in a Controlled Data Set

original plan: https://github.com/brando90/explaining-emergence-icl-in-fms-with-diversity/blob/main/notes/experiment%20plans/_plan_div_performance_ginc.md

# Evaluating the Impact of Data Quality and Diversity on Learning Language Models: A Synthetic Approach

Brando Miranda, contact email: brando9@stanford.edu

# Project Description
In the evolving domain of Large Language Models (LLMs), the quality and diversity of training data play pivotal roles in determining model performance across various benchmarks. 
This project aims to systematically explore the influence of data diversity on enhancing test performance for both specific and general benchmarks, utilizing a controlled synthetic dataset, GINC (Generative In-Context learning Dataset), designed to mimic NLP datasets. 
By creating Synthetic Specific Benchmarks and General Benchmarks, we intend to investigate the hypothesis that diversity significantly improves performance on general benchmarks, potentially expanding the convex hull and facilitating the benefits of transfer learning. 
Additionally, we will delve into the implications of formal quantitative measures of diversity, such as the Task2Vec diversity coefficient, on test performance. 
This exploration is crucial in the current landscape of LLMs like GPT-4, where data-centric tools that automate the pipeline of data selection for improved data efficiency are paramount, potentially breaking existing scaling laws.

# Recommended Background
Candidates are encouraged to share their background when reaching out. A robust foundation in Python is crucial for direct contributions to the project.

Key citations:
- Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data: https://arxiv.org/abs/2306.13840
- The Curse of Low Task Diversity: On the Failure of Transfer Learning to Outperform MAML and Their Empirical Equivalence: https://slideslive.com/38996684/the-curse-of-low-task-diversity-on-the-failure-of-transfer-learning-to-outperform-maml-and-their-empirical-equivalence?ref=search-presentations-low+diversity
- Task2Vec: Task Embedding for Meta-Learning https://arxiv.org/abs/1902.03545
- Beyond neural scaling laws: beating power law scaling via data pruning: https://arxiv.org/abs/2206.14486

# Prerequisites / Preparation
Participants should be comfortable coding in Python and are expected to make substantial contributions to the project. A belief in the paramount role of data and an interest in data-centric ML will be advantageous.

Read my collaboration expectations and send a response to me: https://brando90.github.io/brandomiranda/prospective-collaborations.html

# Motivation
This endeavor is driven by the aspiration to unravel the intricate dynamics between data quality and model performance, contributing to advancements in AI/ML and fostering a deeper understanding of the mechanisms underlying effective model training. By embarking on this journey, we hope to illuminate the pathways to more efficient and robust language models, unlocking new possibilities in the realm of artificial intelligence.

## Goal

Essential Goal: Does pre-training on a highly diverse data set lead to high performance on pre-training evaluation sets?
(note: it's more of a causation than a correlation experiment).

## Experiment Plan 1: Concepts in pre-train set intercepting concepts in test set

1. Fix an eval synthetic GINC data set with 10K concepts (~0.024 diversity coeff. with probe network ... TODO1: what is the probe network? whatever ginc alycia used should be fine),
denote it as `C_{test, 10k, ginc}` and generate a data set `D_eval = D_{test, 10k, ginc}` with `n= ?` examples (TODO2 same as alycia's, or original ginc or something that seems reasonable, if I had to guess at least 30 due to CLT but would choose 500 or a sample complexity guess from learning MHHMs or ask Michael original ginc author)`. 
2. Sample `k <= 10K` concepts from `C_{test, 10k, ginc}` to generate a pre-training data set `D_{tr, k, ginc}` using concpets `C_{tr, k, ginc}`.
3. Compute the diversity of the data set of the previous step and denote it as `div_k = div(D_{tr, k, ginc})`.
4. Pre-train a sufficiently large (e.g., deep) transformer model (TODO3: whatever alycia, michael used for ginc, I think it's a custom GPT/decoder model? I strongly recommend a decoder model)
5. Now start plotting 
   6. A (Ess): `x-axis = div_k vs y-axis = performance_i(D_eval)` (tests main hypothesis/essential goal)
   7. B (Ess): `x-axis = align(D_k, D_eval) vs y-axis = performance_i(D_eval)` (sanity check, tests `if as the alignment in pre-train & test set increases ==> increase in performance`)
   8. C (Ess): `x-axis = div_k  vs align(D_k, D_eval)` (sanity check, `if the diversity increases then ==> increases probability of covering test sets which ultimately ==> increases alignment`)
   9. (D (extra): `x-axis = delta in alginment(D_k', D_eval) vs performance_i(D_eval)` (hypothesis, does the most aligned data added cause the most increase in performance?))
   10. (E (extra): `x-axis = div vs ground truth div` (sanity check div correlates with ground truth div))

Then repeat from step 2 but with a different `k` until `k==10k`. 
Alignment is how aligned is the pre-train & test sets with formulas 
- `alg1(D_k, D_eval) = 1 - div(D_k, D_eval) = 1 - E_{B1 ~ D_k, B2 ~ D_eval}[d_cosine(e_{B1}, e_{B2})]`
- `alg2(D_k, D_eval) = d_cosine(e_{D_k}, e_{D_eval})`

Where `e_{D}` is the Task2Vec embedding (diagonal of the FIM of probe network).

Hypothesis being tests, Rational:
- First let's validate/falsify that if the train set has latent concepts samples exactly from the concepts from the test set, if the eval. performance increases. 
  - Sanity check: This is the simplest scenario, if this doesn't work, it might still be worth sampling different concepts to generate the data set, but I wouldn't expect the latter to work, right?
- test if diversity increase ==> performance eval 
- Sanity Check: as `div increases ==> alignment increases`?

Assumptions/Risk:
- Assumption 1: when plotting `performance_i(D_eval)` we will choose at least 3 metrics `i. ppl, ii. token edit distance (or avg token match), iii. extract string match`
- Assumption 2: the model is **sufficiently large** so that even if the diversity increases by "too much", it won't catastrophically forget or new knowledge would interfere with past knowledge.
- Assumption 3: using performance in the y-axis is enough to detect difference (statistical significant)
  - if we can't detect a difference, perhaps we can use effect size in the y-axis?
- Assumption 4: the code alycia has is easy to run + the architecture they have will be easy to train. 
  - wonder if EasyLM (llama v1 or v2 arch would be better?). Ask easy LM when they will include the better llama v2 arch
  - ask on Twitter why llama v2 trains so stably

Comments:
- note: even though the concepts intersect, the sequences are being generated independently (e.g., with a different seed), so there isn't a contamination from the pre-train set to the test set.
- note: I prefer token edit distance vs avg token match, it's similar to average token error but it's more accepted in the NLP literature (TODO: recall from Rylan why it's a good metric).
- note: **comparisons must be fair** e.g., effectively we must only change the data set (and thus diversity) in the experiments when comparing the performance of different methods. 
  - I suggest we fix: 1. arch 2. compute FLOPS (TODO4: get Rylan's formula) 3. tokens trained on/iterations 4. optimizer 5. anything else?
- not essential but if the experiments work it would be nice to have the diversities normalized in this way: `div'_k = (div_k - min_{D} div(D)) / max_{D} div(D)` i.e., center according to the lowest div divide by the largest diversity.

Decision & justifications for TODOs:
- TODO1 Ans: 
- ...

## Experiment Plan 2: Concepts in pre-train set not necessarily intercepting with concepts in test set
Essentially the same set of experiments as in `Experiment Plan 1` but step 2 changes to adding some `k'` new concepts to the current pre-training data set that not deliberately sampled from the set of concepts in the test sets.
Therefore, we are adding new random concepts to the pre-training mixture.

1. Fix an eval synthetic GINC data set with 10K concepts (~0.024 diversity coeff. with probe network),
denote it as `C_{test, 10k, ginc}` and generate a data set `D_eval = D_{test, 10k, ginc}` with `n = 500?` examples. 
2. Sample `k'` new concepts (not necessarily from `C_{test, 10k, ginc}`) to generate a new pre-training data set `D_{tr, k+k', ginc}` using concepts `C_{tr, k+k', ginc}`.
3. Compute the diversity of the data set of the previous step and denote it as `div_k = div(D_{tr, k, ginc})`.
4. Pre-train a sufficiently large (e.g., deep) transformer model
5. Now start plotting 
   6. A (Ess): `x-axis = div_k vs y-axis = performance_i(D_eval)` (tests main hypothesis/essential goal)
   7. B (Ess): `x-axis = align(D_k, D_eval) vs y-axis = performance_i(D_eval)` (sanity check, tests `if as the alignment in pre-train & test set increases ==> increase in performance`)
   8. C (Ess): `x-axis = div_k  vs align(D_k, D_eval)` (sanity check, `if the diversity increases then ==> increases probability of covering test sets which ultimately ==> increases alignment`)
   9. (D (extra): `x-axis = delta in alginment(D_k', D_eval) vs performance_i(D_eval)` (hypothesis, does the most aligned data added cause the most increase in performance?))
   10. (E (extra): `x-axis = div vs ground truth div` (sanity check div correlates with ground truth div))

Comments:
- note: as we include new data `k'`, we can compute the distance from the test set and see if the more aligned the pre-trained data we train on (add) if that predicts/causes the most increase in performance.
  - note: I wonder if we can compute how far the pre-train and test/eval sets are using (normalized?) MSE/NED of MHMMs ground truth params or hellinger distance
    - we could even provide more data for `div vs ground truth div` ! (we don't even need ground truth to be normalized, we can for visualization purposes)

### Random thoughts
Risk:
- Risk: our eval sets are bad. Tim D. mentiond to me MMLU sucks. Let's ask him why he thinks that, document it and ask him/consider using something else (?). Note, if we only have to evaluate then perhaps it's not too expensive/hard to compute eval on MMLU? 

Comments:
- note: for the real data set, let's articulate which evaluation data sets we are using and why. I propose we use at least 2 from the hugging face eval board: ARC, MMLU.
- note: make sure we know what type of eval we are doing. TODO: ask Brando to share ref. on the subtleties on evaluating LLMs e.g., HELM, etc. let's try to summarize it. 
