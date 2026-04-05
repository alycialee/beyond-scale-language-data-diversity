# Master: Suggested Claude Code Prompts to Address OpenReview Criticisms

**Paper:** "Beyond Scale: the Diversity Coefficient as a Data Quality Metric for Variable Length Natural Language Datasets"
**Authors:** Brando Miranda, Alycia Lee, Sanmi Koyejo
**Submissions:** ICML 2023 Workshop (accept), DMLR@ICLR 2024 (accept), ICLR 2024 (reject), ICLR 2025 (reject)

---

## Recurring Issues Across All Reviews

| # | Category | Severity |
|---|----------|----------|
| 1 | Only 2-3 unrepresentative pre-training datasets (PubMed, USPTO) | Critical |
| 2 | No comparison to baselines (Vendi Score, N-gram, mean embedding) | Critical |
| 3 | Inconsistent diversity→performance correlation not addressed | Critical |
| 4 | Cross-entropy only evaluation; no downstream task benchmarks | High |
| 5 | Confounding factors when merging datasets not controlled | High |
| 6 | Task2Vec methodology underexplained in main text | High |
| 7 | Uninterpretable absolute metric values; bounds claimed "theoretical" are empirical | Medium |
| 8 | Missing recent related work (D4, FineWeb, DataComp-LM, Vendi Score) | Medium |
| 9 | Overclaiming novelty ("paradigm shift"); prior diversity literature not acknowledged | Medium |
| 10 | Presentation: figures too small, unclear referents, citation formatting, long captions | Low |

---

## Suggested Claude Code Prompts

Each prompt below is self-contained and can be pasted directly into Claude Code (`clauded`).

---

### PROMPT 1 — Expand Pre-training Dataset Experiments

```
# Task: Expand diversity coefficient experiments to more representative pre-training datasets

**Context:** The paper "Beyond Scale" (beyond-scale-language-data-diversity/) currently only evaluates
the diversity coefficient on PubMed and USPTO datasets. Reviewers at ICLR 2024 and ICLR 2025 flagged
this as critically unrepresentative. Real LLM pre-training datasets (C4, OpenWebText, The Pile,
RedPajama, SlimPajama, FineWeb, Dolma, DCLM) were not included.

**Goal:** Add diversity coefficient measurements for at least 5 of the following datasets and report
them in Table 1:
- C4 (already partially there)
- OpenWebText / OpenWebText2
- The Pile (full + individual subsets)
- RedPajama or SlimPajama
- FineWeb or FineWeb-Edu
- Dolma

**Steps:**
1. Read the existing experiment scripts in `experiments/` and `src/` to understand how the diversity
   coefficient is currently computed and how results are logged.
2. Check which datasets are already downloaded or easily accessible via HuggingFace datasets library.
3. Write a new experiment script (or extend existing ones) to compute the diversity coefficient over
   the new datasets using the same GPT-2 probe and FIM-based Task2Vec setup.
4. Run on a free GPU on ampere8 (check with:
   `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' '$2+0 < 1000 {print "GPU "$1" free"}'`)
5. Update Table 1 in the paper LaTeX with the new results.
6. Add a brief discussion comparing domain-specific (PubMed, USPTO) vs. general (C4, OWT) diversity scores.

**Key constraint:** Use a single fixed architecture and scale (GPT-2 small, or a fixed 1B model) for
all experiments so that results are comparable across datasets.
```

---

### PROMPT 2 — Add Baseline Diversity Metrics (Vendi Score, N-gram, Mean Embedding)

```
# Task: Add baseline diversity metric comparisons to the paper

**Context:** ICLR 2025 reviewers (N6rW, JTBn) flagged that the paper presents no comparison to
alternative diversity metrics. Specifically missing:
- Vendi Score (Friedman & Dieng, 2023) — entropy-based diversity using kernel matrices
- N-gram diversity (simple token-level diversity measure)
- Mean embedding cosine distance (average pairwise distance in a fixed embedding space)

**Goal:** Implement these three baselines and add a comparison table or figure showing how they
rank the same datasets relative to the Task2Vec diversity coefficient.

**Steps:**
1. Read `src/diversity/` (or equivalent) to understand the current diversity coefficient implementation.
2. Implement Vendi Score: `pip install vendi-score` or implement from scratch using the paper formula.
3. Implement N-gram diversity: for each dataset batch, compute type-token ratio or distinct-n.
4. Implement mean embedding diversity: use a fixed sentence-transformer (e.g. `all-MiniLM-L6-v2`)
   to embed text chunks, then compute mean pairwise cosine distance within a batch.
5. Run all four metrics on the same set of datasets (at minimum: PubMed, USPTO, C4, Wikitext-103,
   The Pile or a subset).
6. Create a comparison table showing rankings and absolute values across metrics.
7. Add a section in the paper (Appendix C or new Section 4.X) discussing where Task2Vec diverges from
   simpler baselines and why the Task2Vec approach is preferable (or not).

**Expected outcome:** Reviewers need to see that simpler baselines fail to capture something that
Task2Vec does, or that Task2Vec agrees with simpler methods (which also validates it).
```

---

### PROMPT 3 — Add Downstream Task Benchmarks as Evaluation

```
# Task: Replace/supplement cross-entropy evaluation with downstream task benchmarks

**Context:** Reviewers at both ICLR 2024 and ICLR 2025 criticized that the only evaluation metric
is cross-entropy / perplexity on C4 and OpenWebText. This is a weak proxy for "data quality."
Reviewers asked for QA, GLUE, or other NLP benchmarks as the ground truth for "quality."

**Goal:** Train small models (GPT-2 small or TinyLlama 1B) on datasets with varying diversity
coefficients, then evaluate on standard downstream benchmarks using lm-evaluation-harness.

**Steps:**
1. Check `~/lm-evaluation-harness/` — it should be installed. If not:
   `pip install lm-eval` or clone from `EleutherAI/lm-evaluation-harness`.
2. Identify which small-scale training runs already exist in `experiments/` and which checkpoints
   are saved (check `checkpoints/` or W&B runs at https://wandb.ai/brando-su/).
3. For each training checkpoint (at minimum PubMed-trained, USPTO-trained, PubMed+USPTO-trained):
   ```bash
   lm_eval --model hf --model_args pretrained=<checkpoint_path> \
     --tasks arc_easy,hellaswag,winogrande,lambada_openai \
     --device cuda:0 --batch_size 16
   ```
4. Create a figure showing: x-axis = diversity coefficient of training data, y-axis = downstream
   benchmark score (average of arc_easy, hellaswag, etc.).
5. Report in the paper alongside or instead of the perplexity results.

**Note:** If training new models is too expensive, use the existing checkpoints and run eval only.
The key claim is: higher diversity coefficient → better downstream benchmark score.
```

---

### PROMPT 4 — Control for Confounding Factors in Dataset Mixing

```
# Task: Add controlled ablations for confounders in mixed-dataset experiments

**Context:** Reviewers (N6rW, Bhvz, Z6o3) pointed out that when PubMed and USPTO are combined,
the performance improvement may come from:
(a) higher diversity, OR
(b) larger dataset size, OR
(c) domain similarity to the evaluation set (C4/OWT), OR
(d) better coverage of evaluation vocabulary

The current paper does not ablate these factors.

**Goal:** Design ablation experiments that isolate the diversity effect from size, domain, and
vocabulary effects.

**Steps:**
1. Read Section 3.1 and Appendix G of the paper to understand the current mixing experiments.
2. Design ablation 1 — Size control: subsample PubMed+USPTO to the same number of tokens as
   PubMed alone, then compare performance. If diversity still wins, the size confounder is addressed.
3. Design ablation 2 — Domain control: find two datasets with similar domain distribution but
   different diversity scores (e.g., two subsets of The Pile). Compare performance.
4. Design ablation 3 — Vocabulary overlap: compute vocabulary overlap between training set and
   evaluation set (C4/OWT) for each training dataset. Report this alongside diversity scores.
5. Add a confounders section to the paper (Section 3.X or Appendix) discussing these ablations.

**Key output:** A table or figure showing that after controlling for size and domain, the diversity
coefficient still predicts performance. Or, if it doesn't, be honest about the limitations.
```

---

### PROMPT 5 — Improve Task2Vec Description and Methodology Clarity

```
# Task: Rewrite the Task2Vec methodology section for clarity and completeness

**Context:** Reviewers JTBn and Bhvz at ICLR 2025 flagged that Task2Vec is underexplained in the
main paper. Reviewer FQiZ at ICLR 2024 raised specific ambiguities:
- What does "partially fine-tuning" mean?
- Is the variable t explicitly defined?
- Is model weight updated after each batch's Task2Vec embedding?
- How is the expectation over both t and x̂_t taken?

**Goal:** Rewrite Sections 2.1 and 2.2 of the paper to fully and unambiguously describe Task2Vec
and the diversity coefficient computation.

**Steps:**
1. Read the current paper LaTeX: find the main `.tex` file in the repo root or `paper/` directory.
2. Read the original Task2Vec paper (Achille et al. 2019) to understand the FIM-based embedding.
3. Rewrite Section 2.1 to include:
   - A precise definition of "partial fine-tuning" (which layers, how many steps, learning rate)
   - Explicit definition of variable t (sequence position)
   - Clear statement of whether model weights are frozen or updated between batches
   - The expectation formula written out fully: E_{t, x̂_t}[...]
4. Add a small pseudocode block (Algorithm 1) showing the diversity coefficient computation
   end-to-end: input dataset → sample batches → compute FIM embeddings → compute cosine distances →
   average → output diversity coefficient.
5. Ensure Figure 1 (or replace it) conveys the computation pipeline clearly with readable font size.
6. Replace "partially fine-tuning" with precise language throughout the paper.
```

---

### PROMPT 6 — Address Inconsistent Diversity→Performance Correlation

```
# Task: Honestly address cases where higher diversity does not improve performance

**Context:** Reviewer Bhvz (ICLR 2025) specifically flagged: "The claim that higher diversity leads
to better performance (Figure 2) does not hold consistently — in some cases performance declines
with increased diversity, which the authors fail to address."

The meta-reviewer also noted: "the correlation between high diversity and the perplexity is
inconsistent where in several cases, the language model performance declines with increased quality."

**Goal:** Add honest analysis of when and why the diversity→performance relationship breaks down.

**Steps:**
1. Read Section 3.1 and all figures in the paper to identify which specific data points show
   diversity increasing while performance stays flat or degrades.
2. Write a new subsection "Limitations and Failure Modes" discussing:
   - The non-monotonic nature of diversity→performance (random tokens = max diversity, bad performance)
   - The specific cases in our experiments where the relationship does not hold
   - Hypotheses for why (domain mismatch, insufficient scale, diversity metric not capturing
     relevant variation)
3. Change any absolute claims ("higher diversity always leads to better performance") to hedged
   claims ("higher diversity tends to correlate with better performance in our experiments, with
   exceptions that suggest...")
4. If data allows, produce a scatter plot of (diversity coefficient, downstream performance) across
   all available checkpoints to show the distribution of the relationship — not just the average trend.

**Note:** Being honest about limitations strengthens the paper. Reviewers are more likely to accept
a paper that acknowledges its limitations than one that ignores contradictory evidence.
```

---

### PROMPT 7 — Update Related Work with Recent Data-Centric Papers

```
# Task: Update the related work section with missing recent citations

**Context:** Reviewers at both ICLR 2024 and ICLR 2025 flagged that the paper is missing many
relevant recent papers. Specific papers mentioned:

From ICLR 2024 reviews:
- D4: Improving LLM Pretraining via Document De-Duplication and Diversification (Meta AI, 2023)

From ICLR 2025 reviews (Bhvz):
- Multiple papers from ~2023-2024 on data diversity and data-centric ML (7 missing citations listed
  in the review)

Also implied missing:
- Vendi Score (Friedman & Dieng, 2023)
- FineWeb (Penedo et al., 2024)
- DataComp-LM / DCLM (Li et al., 2024)
- DoReMi (Xie et al., 2023)
- Data selection literature (LESS, LIMA, etc.)

**Steps:**
1. Fetch the ICLR 2025 review from https://openreview.net/forum?id=kDakBhOaBV and copy the full
   list of missing citations from reviewer Bhvz.
2. Search for each paper on arXiv or Semantic Scholar to get BibTeX entries.
3. Read the current `related_work` section in the paper LaTeX.
4. Add a new paragraph or extend the existing one to cover:
   - Data deduplication and diversification methods (D4, SemDeDup)
   - Data selection methods (LESS, LIMA, DoReMi, DataComp)
   - Quality filtering methods (FineWeb, RefinedWeb, DCLM)
   - Other diversity metrics (Vendi Score, n-gram diversity)
5. For each new citation, add 1-2 sentences explaining how our work relates to or differs from it.
6. Update the .bib file with all new entries.
```

---

### PROMPT 8 — Fix All Presentation and Formatting Issues

```
# Task: Fix all presentation, formatting, and clarity issues flagged by reviewers

**Context:** Reviewers Bhvz (ICLR 2025) and H6fT (ICLR 2024) gave detailed lists of formatting
and clarity issues. These are low-hanging fruit that can be fixed without new experiments.

**Steps:**
1. Find the main LaTeX file and open it.
2. Fix citation style: change `\cite{}` to `\citep{}` for parenthetical citations (L.33 and throughout).
3. Define "latent concepts" (L.66) or remove the term and replace with a more precise description.
4. Fix "by them to" (L.69) grammatical error.
5. Fix formula at L.122: "t-1:1" should be "1".
6. Fix "pre-trained on the English language" (L.151) — rewrite to be more precise.
7. Shorten all figure and table captions — move details to main text.
8. Clarify argument about lower bound (L.191-192).
9. Define "non-special" token (L.197).
10. Define or replace "formal diversity" (L.209).
11. Break Figure 3 into subfigures (a), (b), (c), (d).
12. Clarify notation "+0.03-0.05" (L.317) — use ± or explicit range.
13. Move Section 3.4 to the beginning to establish metric validity early.
14. Fix "top" → "right/left" (L.360).
15. Increase all figure font sizes so they are readable without zooming.
16. Fix all unclear pronoun referents ("by them", "they ...") throughout.
17. Add a mention of the data used to pre-train GPT-2 in Section I.2 / Appendix.
18. In L.751: replace "this more sophisticated aggregation method" with the explicit method name.

**After fixing:** Compile the LaTeX and do a full read-through checking for any remaining issues.
```

---

### PROMPT 9 — Add Interpretability / Human Annotation Validation

```
# Task: Add human annotation study to validate the diversity coefficient

**Context:** Reviewer v5Te (ICLR 2025) said: "The experiments on synthetic datasets are valuable,
however the conclusions could be further strengthened by a study utilizing human annotation
for diversity."

The absolute values of the diversity coefficient (range ~0.05–0.4) are not intuitive to readers.
A human annotation study would:
1. Validate that the metric agrees with human judgment of diversity
2. Make the metric interpretable (e.g., "score of 0.3 corresponds to what humans would rate as X")

**Steps:**
1. Sample 20-30 pairs of text batches from datasets with different diversity scores.
2. Design a simple annotation task: given two batches of text, which batch is more diverse?
3. Recruit 3-5 annotators (lab members are fine for a small study).
4. Compute inter-annotator agreement (Cohen's kappa).
5. Report: what fraction of the time does the diversity coefficient agree with majority human judgment?
6. Add this as a new validation experiment (Section 3.X) in the paper.

**Alternative (if human annotation is too slow):** Use GPT-4 as an annotator — present pairs of
text batches and ask it to judge which is more diverse. Compare its judgments to the coefficient.
This is faster and still adds validation beyond purely computational metrics.
```

---

### PROMPT 10 — Address "Paradigm Shift" Overclaiming and Missing Diversity Literature

```
# Task: Tone down overclaiming and properly situate the work in prior diversity literature

**Context:** Reviewer Bhvz (ICLR 2025) flagged: "The authors assert their work represents a
'paradigm shift' in data-centric ML through the introduction of data diversity (L. 59) — this is
factually incorrect, as many previous studies have explored data diversity."

**Steps:**
1. Search for prior work on data diversity in NLP/ML — specifically papers before 2023 that:
   - Measure dataset diversity
   - Study the effect of diversity on model performance
   - Propose diversity metrics for text data
2. Rewrite L.59 and surrounding text to:
   - Acknowledge prior diversity work
   - Clearly articulate what is NEW about the Task2Vec / diversity coefficient approach
   - Replace "paradigm shift" with a more measured claim (e.g., "we propose the first
     Task2Vec-based diversity coefficient for variable-length NLP datasets, extending prior
     work on...")
3. Add a "What is novel" paragraph in the introduction that explicitly contrasts this work with:
   - Vendi Score
   - N-gram diversity
   - Prior Task2Vec work (Achille et al.)
   - Data selection / filtering papers
4. Ensure the contributions list in the introduction is specific and defensible.

**Goal:** A reviewer should finish the introduction knowing exactly what is new, what builds on prior
work, and why the new parts matter — without feeling that the authors are overselling.
```

---

## Priority Order for Addressing Reviews

| Priority | Prompt | Impact on Acceptance |
|----------|--------|---------------------|
| 1 | PROMPT 1 — Expand datasets | Critical — directly addresses weakest empirical claim |
| 2 | PROMPT 2 — Add baselines | Critical — missing baselines is a standard rejection reason |
| 3 | PROMPT 6 — Address inconsistencies | Critical — must explain failure modes |
| 4 | PROMPT 3 — Downstream benchmarks | High — PPL is insufficient; ARC/HellaSwag etc. are standard |
| 5 | PROMPT 4 — Confounders | High — science requires ablations |
| 6 | PROMPT 5 — Task2Vec clarity | High — reviewers can't evaluate what they can't understand |
| 7 | PROMPT 7 — Related work | Medium — citation gaps undermine credibility |
| 8 | PROMPT 10 — Tone down claims | Medium — overclaiming alienates reviewers |
| 9 | PROMPT 8 — Presentation fixes | Low — easy wins, do last |
| 10 | PROMPT 9 — Human annotation | Low — nice to have, not blocking |
