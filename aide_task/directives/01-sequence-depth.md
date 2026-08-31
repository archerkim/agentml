**Direction: user behavioural history, taken much further than "one previous item".**

What is already known (do not just re-derive it): a single `prev_video` field is worth
about +0.002 valid primary over the FM baseline (run 10), and `user_recent_author`
about the same (run 9). "History as separate embedded fields" beat DIN-style attention
in earlier work (test 0.5981 vs 0.5949).

Go deeper than that this run:
- Represent K previous interactions, not one. Sweep **K ∈ {5, 10, 20, 50}** explicitly
  and report the validation primary for every K you try.
- Try at least two pooling schemes for the history: mean-pooling the embeddings, and
  keeping the most recent few as separate fields. Report both.
- Consider histories built from different signals: long_view positives, is_click, and
  all impressions regardless of outcome. These are different sequences; say which you used.
- Keep the construction causal and split-aware (train history for valid, train+valid for
  test). A non-causal history silently leaks and has already destroyed one candidate
  this session.

Report a small table of (K, pooling, source) -> valid primary in your output.
