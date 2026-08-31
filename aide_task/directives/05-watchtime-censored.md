**Direction: censored watch-time modelling as an auxiliary signal.**

What is already known: using a censored watch-time regression DIRECTLY as the ranking
score failed badly (test 0.5516). That is a recorded dead end — do not repeat it. What
has never been tried is using watch time as an AUXILIARY objective alongside the
long_view classification head, which is the actually promising form of this idea.

- `play_time_ms` and `duration_ms` are available in train/valid (play_time_ms is NOT in
  test — training-time target only). Real watch time is censored: a video that ends
  before the user would have stopped truncates the observation, so a one-sided/censored
  loss is appropriate rather than plain MSE.
- Train long_view as the primary head and censored watch-time as a secondary head over a
  shared representation. Rank by the long_view head.
- Sweep the auxiliary weight **∈ {0.1, 0.3, 1.0}** and the completion-ratio threshold used
  to define censoring **∈ {0.9, 1.0}**. Report valid primary for each.
