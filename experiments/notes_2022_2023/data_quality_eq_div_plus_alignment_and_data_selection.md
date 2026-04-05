# Data Quality as Data Diversity/Coverage + Data Alignment

Goal: Design a robust & thorough metric for data quality that causes better data selection and thus better data.

Note: we propose to change cross diversity coefficient to be called now the coverage coefficient. 

# Proposal
Recall essential goal: design a metric that increases test performance. 

Motivation:
- You want your Source task to intersect as much as possible with the target domain.
- Thus, the goal should be:
  - 1. have source domain **align** with the target domain
  - 2. once aligned, **cover** as much as possible the range of the target domain

Analogy: data quality tries to center the source domain with the "mean task" of the target domain and then try to cover/encapsulate the variance of the whole target domain.
So have the "mean" task in the source align with the "mean" task in the target.
Then have the spread/variance of the source align and cover the spread/variance of the target domain.

With that in mind we propose the data quality metric DQ given a (set) of source domains and a (set) of target domains as follows (usually represented as data sets):
```
DQ(S, T) = alpha_{alg} * "Align mean Task Source with Target" + alpha_{coverage} * "Cover Target's Task Spread with Source"
DQ(S, T) = alpha_T * Alignment(S, T) + alpha_S * Diversity(S, T)

S := Source (e.g., train data set, e.g., C4 or C4 + USPTO)
T := Target (e.g., test data set, e.g., test OpenWebText or domain specific e.g., IsaProofNet for Autoformalization or maths textbooks)
```
More concretely
```markdown
# Data Quality := alignment + (cross) diversity
DQ(S, T) = alpha_T * Alingment(S, T) + alpha_S * Coverage_Coeff(S, T)

# 1. first, center S -> T (so use large batch size for both, e.g., whole data set or 1024, 2028)
# 2. second, match S's spread to T (for computation & avoiding OOM/memory issues use smaller batch size, seq length e.g., 512 as in beyond scale)

DQ(S, T) = alpha_a * E_{B_{S, inf} ~ S, B_{T, inf} ~ T}[ 1- d( e_{B_{inf, S}}, e_{B_{inf, T} } ) ] + alpha_c * E_{B_{S, 512} ~ S, B_{T, 512} ~ T}[ d( e_{B_{512, S}}, e_{B_{512, S} } ) ]

note: due to inf sample for alignment, we make fewer sampler for aligment. If exactly inf then a single sample (ideal)

S := Source (e.g., train data set, e.g., C4 or C4 + USPTO)
T := Target (e.g., test data set, e.g., test OpenWebText or domain specific e.g., IsaProofNet for Autoformalization or maths textbooks)
d := cosine distance
e_{B} := Task2Vec embeddings
B_{B, N} := batch of data currently sampled (~ task) from domain B with N examples.
alpha_i := mixing coeffs (tbd how to remove HP or choose it well...DoReMi? Or good Hps that works in different settings or specific domains e.g., AF)
```
We say "the data quality of Source Domain to Target Domain (S2T).

Thoughts/tips:
- use consistent setting of probe network & hps to make DQ comparable. e.g., values from GPT2 vs LLaMAv2 might not be comparable
- using e := T2VEmbds to embed batches/tasks makes our metrics **multi-modal**.
- Note, although coverage coeff and alignement coeff seem similar (i.e., corss_div() , 1-cross_div()), due to different batch sizes for ((very non-linear** Task2Vec embedding method, it results in different values and not just `a_1 corss_div() +a_2 1-cross_div() = 1 - (a_1 - a_2)corss_div()`

##

TODO: display equations in markdown https://chat.openai.com/c/f43e625b-dbd6-4ecc-93f6-d00f9aa10af6 

### References

Overleaf: https://www.overleaf.com/read/wbfqfshkcgbq#4b2dc1
