# Data Quality as Data Diversity + Data Alignment

Goal: Design a robust & thorough metric for data quality that causes better data selection and thus better data.

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

S = Source (e.g., train data set, e.g., C4 or C4 + USPTO)
T = Target (e.g., test data set, e.g., test OpenWebText or domain specific e.g., IsaProofNet for Autoformalization or maths textbooks)
```
More concretely
```markdown
# Data Quality := alignment + (cross) diversity
DQ(S, T) = alpha_T * Alingtment(S, T) + alpha_S * Cross_Diversity(S, T)
# 1. first, center S -> T
# 2. second, match S's spread to T

DQ(S, T) = alpha_T * E_{B_{S, inf} ~ S, B_{T, inf} ~ T}[ d( e_{B_{inf, S}}, e_{B_{inf. S} } ) ] +
alpha_S * E_{B_{S, inf} ~ S, B_{T, inf} ~ T}[ 1 - d( e_{B_{512, S}}, e_{B_{512. S} } ) ]

```


TODO: display equations in markdown https://chat.openai.com/c/f43e625b-dbd6-4ecc-93f6-d00f9aa10af6 
