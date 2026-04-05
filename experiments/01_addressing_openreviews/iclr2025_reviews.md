# ICLR 2025 Submission Reviews — Beyond Scale

**Forum:** https://openreview.net/forum?id=kDakBhOaBV
**Decision:** Reject
**Paper:** "Beyond Scale: the Diversity Coefficient as a Data Quality Metric for Variable Length Natural Language Datasets"
**Authors:** Brando Miranda, Alycia Lee, Sanmi Koyejo

---

## Meta-Review (Area Chair)

> "Reviewers agree that this is an important problem to study... However, the empirical evidence for the correlation between diversity and performance is limited, particularly regarding the choice of datasets as they might not reflect realistic pretraining datasets, limited number of datasets, and potential confounders combining different datasets might have introduced. The reviewers also point out that the correlation between high diversity and the perplexity is inconsistent where in several cases, the language model performance declines with increased quality. Furthermore, the paper lacks a comparison with other diversity metrics or simpler embedding-based approaches. Based on the reviewers' criticism, I recommend reject at this time."

---

## Reviewer v5Te — Rating: 5 | Soundness: 3 | Presentation: 2 | Contribution: 3 | Confidence: 4

### Weaknesses
1. **Motivation for cosine distance**: The use of cosine distance between Task2Vec embeddings as a diversity measure is not well motivated. Other textual embedding methods should be considered; tradeoffs should be discussed.
2. **Uninterpretable absolute values**: The absolute diversity coefficient value does not convey much information — the conceptual lower and upper bounds don't capture any representation of real natural language. The work should present a better way to understand the value of the measure.
3. **Human annotation study missing**: The experiments on synthetic datasets are valuable, but conclusions could be further strengthened by a study utilizing human annotation for diversity.

### Questions
- In section 3.5, the authors use the GINC dataset generated from a mixture of hidden Markov models — how natural is the resulting dataset? Does it come close to resembling text found on the web?

---

## Reviewer JTBn — Rating: 5 | Soundness: 2 | Presentation: 1 | Contribution: 2 | Confidence: 2

### Weaknesses
- **Weak evaluation metric**: While the authors evaluate performance against pre-training data diversity (section 3.1), the metric for performance is cross-entropy loss in LM; specific task evaluation on NLP tasks (QA, GLUE, etc.) would make more sense.
- **Missing Vendi Score baseline**: The Vendi Score seems to be another approach to compute diversity — why was it not included as a baseline in the main experiments (at least in Table 1)?

### Questions
1. Line 122: which $t$ is used to compute the FIM matrix for sequences of a batch? Are all the steps from the sequence averaged or is FIM only built based on the last step?
2. Several pieces of text use phrases like "by them", "they ..." with unclear referents, making reading more involved.
3. Task2Vec should be fully and better described in the main paper (including prior use, its intuition, etc.).
4. Section I.2 should mention the data used to pre-train GPT-2.
5. Figure 1 does not seem to add much; an algorithm would be more useful.
6. All figures are too small to read.
7. Line 751: "this more sophisticated aggregation method" — which aggregation method?

---

## Reviewer N6rW — Rating: 3 | Soundness: 1 | Presentation: 2 | Contribution: 1 | Confidence: 3

### Weaknesses

**On the main claim "PRE-TRAINING IN HIGHER DIVERSITY LEADS TO BETTER EVALUATION PERFORMANCE" (Section 3.1):**
- Only three datasets are considered: PubMed, USPTO, and PubMed+USPTO. Linear regressions on three data points are presented as evidence.
- The actual relationship between diversity and performance will neither be linear nor monotonic — pre-training on random tokens would certainly lead to very bad performance while having maximal diversity. For most GPT-2 experiments, PubMed+USPTO has similar or higher loss than PubMed while also having higher diversity.
- The two pre-training datasets are highly unusual and not representative of current LLM pre-training corpora. More convincing choices: C4, OpenWebText, The Pile, RedPajama, SlimPajama, RefinedWeb, Dolma, FineWeb, DCLM, etc.
- Many small-scale training runs vary scale (parameters, tokens) and architecture (GPT-2 or Llama 2), but what should vary are the **datasets**. Much more convincing if authors considered a single architecture/scale (e.g., Chinchilla-optimal 1B) but as many pre-training datasets as possible.
- The choice of PPL on OpenWebText2 and C4 as evaluation metric is unclear. Does PubMed lead to lower PPL because of its higher diversity, or because it is simply more similar to C4/OpenWebText? Work on pre-training data quality has started using "early-signal benchmarks" (see FineWeb or DataComp-LM papers).

**On other empirical results:**
- It is not particularly compelling to show that concatenating datasets leads to higher diversity, or that the cross diversity coefficient is higher than a single dataset's diversity. It would be interesting to know if simpler approaches (N-gram, mean GPT-2 last layer embedding) fail such checks.
- It is unclear why Task2Vec encodings are preferable over N-gram or vector embedding-based similarity metrics. Appendix C discusses this but should be moved to the main text. Much more compelling if authors replicated core empirical results (Figures 2, 3, 4) with simpler baselines to show their limitations.

---

## Reviewer Bhvz — Rating: 3 | Soundness: 2 | Presentation: 2 | Contribution: 2 | Confidence: 4

### Weaknesses

**Validity of Claims:**
- The authors assert their work represents a "paradigm shift" in data-centric ML through the introduction of data diversity (L. 59) — this is factually incorrect, as many previous studies have explored data diversity. Several relevant papers were overlooked (7 missing citations).
- The claim that the method is "interpretable" is problematic. The metric is as much a black-box as other metrics that output a single number. Additionally, the range of values (lower bound ~0.05, upper bound ~0.4) is not intuitive.
- The claim that higher diversity leads to better performance (Figure 2) does not hold consistently — in some cases performance declines with increased diversity, which the authors fail to address. Presenting results in graphs rather than tables makes it difficult to investigate this trend thoroughly.
- The paper does not adequately account for potential confounding factors. When merging two datasets, other characteristics of the combined dataset may influence performance aside from diversity.
- L.111: The assertion that the probe network's parameters are the most important for solving the task may be incorrect.
- L.266+269: The bounds described as "theoretical" are actually empirical.

**Evaluation:**
- The evaluation is based solely on cross-entropy scores, which is not a strong or convincing measure. Even if all results showed consistent trends (which they do not), the claim that increased diversity leads to better downstream performance is weakly supported.

**Formatting and Clarity:**
- L.33: Missing parentheses around citations (use `\citep`).
- L.66: "latent concepts" is undefined.
- L.69: Grammatical issue with "by them to."
- Figure 1 is too small.
- L.122: In the formula, "t-1:1" should be "1."
- L.151: "pre-trained on the English language" is awkward.
- Captions for tables and figures are unnecessarily long.
- L.191-192: Argument about the lower bound is unclear.
- L.197: "non-special" token needs clarification.
- L.209: "formal diversity" is unclear.
- Figure 3: Should be broken into subfigures (a-d).
- L.317: Notation "+0.03-0.05" needs clarification.
- Section 3.4 seems out of place — should be moved to the beginning to establish validity of the metric early on.
- L.360: "right/left" was likely intended instead of "top."

---

## Summary of Issues (ICLR 2025)

| Category | Issues |
|----------|--------|
| Empirical evidence | Only 3 data points (PubMed, USPTO, mix); non-representative datasets |
| Inconsistency | Higher diversity does not always correlate with better performance |
| Confounders | Merging datasets changes more than diversity; PPL eval may capture domain similarity, not diversity benefit |
| Baselines | No Vendi Score, N-gram, or embedding baselines |
| Evaluation | Cross-entropy only; no downstream task benchmarks (QA, GLUE, etc.) |
| Interpretation | Absolute metric values are uninterpretable; lower/upper bounds are empirical not theoretical |
| Task2Vec description | Insufficient explanation in main text; FIM computation ambiguity |
| Presentation | Figures too small; unclear referents; overclaiming "paradigm shift" |
| Related work | Missing several recent diversity and data-centric papers |
