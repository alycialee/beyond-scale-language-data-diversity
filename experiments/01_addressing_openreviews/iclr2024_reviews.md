# ICLR 2024 Submission Reviews — Beyond Scale

**Forum:** https://openreview.net/forum?id=506Sxc0Adp
**Decision:** Reject
**Paper:** "Beyond Scale: the Diversity Coefficient as a Data Quality Metric for Variable Length Natural Language Datasets"
**Authors:** Brando Miranda, Alycia Lee, Sanmi Koyejo

---

## Meta-Review (Area Chair)

> "A major issue raised in the reviews was around the paper's novelty... The paper largely relies on existing methodologies, including Task2Vec diversity coefficient and latent concept analysis, and does not offer new or noteworthy findings... Another notable issue raised was the need to survey more recent data-centric works. It could be that the novelty issues are a matter of a clearer comparison with previous works and a better explanation of the innovation on top of them, but either way the paper in its current form does not seem to be ready to be published in ICLR."

---

## Reviewer FQiZ — Rating: 1 (Strong Reject) | Confidence: 4

### Weaknesses
1. **Limited Novelty and Insight**: The paper largely relies on existing methodologies (Task2Vec diversity coefficient and latent concept analysis) and does not offer new or noteworthy findings.
2. **Issues with Model Updates and Task2Vec**: It's not clear if model weights are updated after computing the Task2Vec embedding for each batch. This is crucial as it affects the validity of results in Section 4.
3. **Unexplained Necessity for Task2Vec**: The paper uses token distribution as a metric but fails to clarify why Task2Vec coefficients are needed instead of direct token distribution metrics.
4. **Unclear Practical Utility**: The paper suggests the diversity coefficient as a potential data quality metric but does not empirically validate this claim.
5. **Methodological Ambiguities**: Undefined concepts in the methods section.

### Questions
1. Is the model weight updated after the Task2Vec embedding is computed for each batch? If so, diversity should inherently be high since information from the first batch has already been learned — how does this affect results in Section 4?
2. In Section 2.2.4, token distribution is used as a metric for dataset diversity. Why then is it necessary to calculate Task2Vec diversity coefficients by fine-tuning the model?
3. Ambiguities in Section 2.1:
   - (a) What is meant by "partially fine-tuning"?
   - (b) The variable $t$ is not explicitly defined.
   - (c) How is the expectation over both $t$ and $\hat{x}_t$ taken?
4. Potential typo: "...the **high** diversity of public datasets for LLM pre-training is **high**..."

---

## Reviewer kREU — Rating: 3 (Reject) | Confidence: 4

### Weaknesses
- **Unclear practicality**: The metric is shown to reasonably estimate inherent "diversity in a dataset," but its extrapolation to measuring "dataset quality" is unsupported.
- **Limited novelty**: The paper directly builds on two existing lines of work (Miranda et al. on Task Diversity and Task2Vec), and is limited in application to different practical scenarios.
- **Model dependence**: The proposed diversity metric is model-dependent (GPT-2 is used), while the downstream effect of the probe model on the diversity metric is not studied.

### Questions
- What are some practical use-cases for the proposed diversity metric?
- Why do you think data diversity should be a good indicator of data quality?
- A larger batch-size in addition to providing diverse data also directly affects optimization — how does model performance change with increasing batch size?
- Would data diversity estimated using GPT-2 as the probe-model translate to better training of (1) models with different architectures (encoder-decoder, encoder-only), and (2) different model-sizes (GPT-3+)?

---

## Reviewer H6fT — Rating: 6 (Marginally Above Threshold) | Confidence: 2

### Weaknesses
1. The definitions in sections 2.1 and 2.2 are somewhat difficult to comprehend; more illustrations should be provided.
2. More experiments should be conducted to investigate the correlation between diversity coefficient and LLM performance — on more models and datasets with multiple runs.

---

## Reviewer Z6o3 — Rating: 6 (Marginally Above Threshold) | Confidence: 4

### Weaknesses
1. **Weak lower bound**: If the lower bound is treated as baseline, it is too weak to say a dataset "having better diversity is really diverse."
2. **Computational overhead**: Since this approach requires fine-tuning an LM, it introduces computational overhead that should be stated in the limitations section.
3. **Missing recent related work**: More recent data-centric related works should be added (most listed works are pre-2022). The paper "D4: Improving LLM Pretraining via Document De-Duplication and Diversification" (Meta AI, 2023) should be cited.
4. **Missing code diversity analysis**: Since coding data is increasingly important, authors should provide more insights about diversity of code pre-training datasets.
5. **Unjustified conclusions**: Appendix G concludes that "higher data diversity leads to higher test performance" but fails to ablate dataset overlap (the Pile contains both USPTO and PubMed, so better performance on USPTO+PubMed cannot purely be attributed to diversity).

### Questions
1. The dimension of Task2Vec embeddings should equal the number of model parameters — if true, the vector will be extremely large. Clarification needed.
2. Does "Randomly initialized GPT-2 without fine-tuning" mean only resetting the LM head or the whole model? Different random initializations could lead to very different results — is it safe to conclude random networks "always underestimate diversity"?
3. What is the relationship between the diversity coefficient and the cross diversity coefficient?
4. In Table 1: why does concatenating C4 and Wikitext-103 **decrease** the Div Coeff?
5. In Table 1: what does "Combination of Five Datasets (MIX2)" mean? Why only five datasets? Why 0.77 vs 0.23?

---

## Summary of Issues (ICLR 2024)

| Category | Issues |
|----------|--------|
| Novelty | Heavy reliance on Task2Vec; no new algorithmic contribution |
| Empirical | 2-3 datasets only; no correlation study with downstream tasks |
| Baselines | No comparison to N-gram, embedding, or Vendi Score metrics |
| Methodology | Ambiguous partial fine-tuning; undefined variables; model weight update unclear |
| Practical utility | No actionable guidance; connection between diversity and quality unvalidated |
| Related work | Missing D4, recent data-centric ML papers |
| Confounders | Dataset overlap not controlled in performance claims |
| Presentation | Unclear definitions; needs more illustrations |
