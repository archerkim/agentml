**Direction: pairwise/listwise ranking loss COMBINED with history features.**

What is already known: pointwise-vs-ranking loss alone is exhausted — six independent
runs with LambdaRank/LGBMRanker land in test 0.591–0.595, no better than baseline. Naive
hard-negative pairwise mining (PDAOM-style) actively hurt (test 0.5885).

But ranking loss has never been combined with the sequence features that DID work. The
task description says explicitly: "the best result will likely come from combining
ranking loss (already proven) with sequence features (new)". That combination is this
run's job.

- Optimise a within-user ranking objective (BPR-style pairwise, or a softmax/listwise
  loss over each user's impressions) rather than pointwise logloss.
- Feed it the history features that already work (previous video/author, K previous items).
- Sweep the loss hyperparameters explicitly: number of sampled negatives per positive
  **∈ {1, 4, 16}**, and learning rate **∈ {1e-3, 3e-3}**. Report every setting's valid primary.
- Avoid hardest-negative mining: it is a recorded dead end here.
