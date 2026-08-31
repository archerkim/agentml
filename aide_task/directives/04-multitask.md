**Direction: multi-task learning with auxiliary engagement labels.**

What is already known: a proper jointly-trained MMoE (shared experts + per-task gates,
scored from the long_view tower only) reached test 0.5964, beating baseline. Blending
separately-trained per-task models post-hoc did NOT work — the tuned blend weight came
out 100%/0%. So the auxiliary tasks must shape a SHARED representation during training,
not be mixed afterwards.

- Train long_view jointly with auxiliary heads on `is_click`, `is_like`, and optionally
  `is_follow`/`is_comment`/`is_forward`. These columns exist in train.csv and valid.csv
  but NOT in test.csv — they may only be training-time targets, never inference features.
- Sweep the auxiliary loss weights explicitly: **∈ {0.1, 0.3, 1.0}** relative to the main
  task, and the number of shared experts **∈ {2, 4}**. Report valid primary for each.
- Score using the long_view head alone.
