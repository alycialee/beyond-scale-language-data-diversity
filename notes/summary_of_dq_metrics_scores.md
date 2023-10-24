# Review of Data Quality Metric

# TODO
so the papers for homework for brando to write a related work + how it relates to my work:
- vendi score [x] but put here
- do re mi [x] put here
- dsrni
- ssl proto
- d4
- skill it
- dedup
- Task2Vec
- Task2Vec Div Coeff & Task2Vec Alg

## General thing to keep in mind
Selecting data must be cheaper than training on it. Otherwise just train on all of it. (data selection should be embarrasingly parallel, no depedencies unlike SGD + consider training has the cost of forward + backward pass)

## SSL Prototype
Method:
- create centroids of the data using k-means (unsupervised or supervised) (unclear if they cluster train or val, I suggest val, then this method is more comparable to my Task2Vec alginment/dq metric)
- score/metric := `ssl_proto(data_point, centroid) = cosine(data_point, centroid)`
  - they define high quality (to trian on) as "hard examples" -- so the least similar to the prototypes (nearly opposite of my alignment metric?)

Pros/Cons of SSL proto
Pros:
- (+) curious how they select data points efficiently given GPUs work better with batches. Maybe the compute metrics in batch then loop through batch dimension to select data points?
Cons:
- (-) not clear they use val set (but easy to fix?)
- (-) I think task2vec (L1) complexity is better (more principled due to kolmogorov)

Differences with our Task2Vec alignment metric:
- We score based of assumption that "training on most similar to test set" makes most sense. If we had the "test set" we'd train on it.
  - based on this, training on the most different examples to your prorotypes doesn't make sense. But this only means that perhaps their definition of hard isn't pricipled/optimal
  - this opens the opportunity to use Task2Vec complexity (L1 of Task2Vec embeddings) -- train on the "Task2Vec embeddings most similar to the target domain (val)". e.g., Task2Vec is related to Kolmogorov complexity, so perhaps this is a motivation to believe this different approach is better.
  - interesting difference. Opens up for different abalations:
    - 1. choose data based on t2v algiment e.g., compute Task2Vec alignment based on large batch of target (val) domain & choose data points with Task2Vec embeddings most similar to to the target embedding (note: we need to figure out how to make Task2Vec more computationally feasible, data parallelism for example, but their method can do that too)
      2. choose data based on t2v algiment + div https://github.com/brando90/beyond-scale-language-data-diversity/blob/main/notes/data_quality_eq_div_plus_alignment_and_data_selection.md
      3. Prototypes constructions: Centroids vs large batch embedding with Task2Vec for measuing t2v aligmnent or SSL proto metric.
      4. testing their SSL proto metric on it's own

Questions: 
- Q1: How do they choose how many points to choose?
  - A1: Not sure, but I assume they score N example (all the train) and then select k based on compute. So k is pre-given by compute budget.
- Q2: Why cosine distance?
  - A2: I think it's due to margin inspiration, but Task2Vec is more principled (imho) and used L1 of Task2Vec embeddings. Space to improve here! 
