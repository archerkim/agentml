# Per-run directives

One file per research direction. `campaign.py` passes each in turn to
`run_with_early_stop.py --directive`, which appends it to the task description.

Why they exist: `findings.jsonl` tells every run what has already been tried, but
nothing told a run where to go *next*. Runs 9 and 10 were seeded identically, had the
full findings file, and independently converged on the same idea ("FM baseline + one
history-derived categorical field") — 0.6035 and 0.6036. Without a directive, extra
runs buy repetition rather than coverage.

Each directive names one direction, states what is already known about it from
`findings.jsonl`, and requires an explicit internal hyperparameter sweep so a single
run reports several settings rather than one point estimate.
