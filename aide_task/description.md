# Task

Rank `long_view` (0/1) **within each user** over KuaiRand-Pure impression logs — for each
user, rank only the videos actually shown to them in the evaluation split (train/valid/test),
not a full-catalog search.

## Data (already prepared — do NOT re-download or rebuild the splits yourself)

- `train.csv` — 20220408–20220421, full log with all columns (including `long_view` and the
  other outcome labels: `is_click/is_like/is_follow/is_comment/is_forward/is_hate/
  play_time_ms/profile_stay_time/comment_stay_time/is_profile_enter`).
- `valid.csv` — 20220422–20220428, same columns. Use for iterative feedback between candidates.
- `test.csv` — 20220429–20220508, **outcome columns are physically stripped** (this is the
  hidden test set). Only `row_id,user_id,video_id,date,hourmin,time_ms,duration_ms,is_rand,tab`
  are present. You will never see `long_view` for these rows — final scoring happens once,
  outside your access.
- `sample_submission.csv` — answer format.
- `log_random_valid.csv` — 20220422–20220428, a slice of the **random** (unbiased) impression
  log, optional for extra validation (a model can overfit to the biased train/valid traffic).
- `user_features_pure.csv`, `video_features_basic_pure.csv` — static user/video sides, not tied
  to a specific impression, safe on all splits.
- `evaluate.py` — the official scorer. **Do not touch this file.** Always call
  `from evaluate import evaluate`; never write your own GAUC/nDCG — any discrepancy in the
  metric implementation breaks comparability with the baseline.

**Working directory layout (important for imports).** Your script runs with a cwd where
`input/` and `working/` are sibling folders (not `./evaluate.py`, but `./input/evaluate.py`).
All the files listed above (including `evaluate.py`, `train.csv`, `valid.csv`, `test.csv`)
live inside `input/`. Working template for the start of your script:

```python
import sys
sys.path.insert(0, "./input")   # must come BEFORE "from evaluate import evaluate"
from evaluate import evaluate
import pandas as pd

train = pd.read_csv("./input/train.csv")
valid = pd.read_csv("./input/valid.csv")
test  = pd.read_csv("./input/test.csv")
# write your submission and other artifacts to "./working/"
```

Without `sys.path.insert(0, "./input")`, `from evaluate import evaluate` will fail with
`ModuleNotFoundError: No module named 'evaluate'` — this is the only cause of that error if
it occurs.
- Full data dictionary and explicit leakage warnings — see `data_dictionary.md` in this same
  folder. **Read it before designing features.** In particular:
  `video_features_statistic_pure.csv` is deliberately not included in data_dir — these are
  per-video aggregates like `long_time_play_cnt`, computed over the entire log including the
  test period; using that file is a direct target leak. If you need a similar signal, compute
  video aggregates yourself, from `train.csv` only.

## Metric (fixed — do not redefine)

`evaluate(user_ids, labels, scores)` returns `GAUC`, `nDCG@5`, `primary = mean(GAUC, nDCG@5)`.
- GAUC: per-user AUC, weighted by number of positives, computed only for users with
  `0 < positives < impressions`.
- nDCG@5: 0 for users with no positives (still included in the average); 1 for users who are
  fully positive.
- Optimize `primary` on `valid.csv`.

**`primary` (like `GAUC` and `nDCG@5`) is always maximize — higher is better.** This is not a
minimize metric under any framing (not an error/loss). When reporting the results of each run
(is_bug/summary/metric/lower_is_better), `lower_is_better` must be `false` in every single
case — the metric direction never changes between candidates in this task.

**How to read the numbers:** the metric is not stretched across `[0, 1]`. 27.1% of users have
zero positives (their nDCG is permanently 0, no model can fix that), and 9.2% are fully
positive. Because of this, even a perfect ranking (oracle, prediction = true label) reaches not
1.0 but `GAUC=1.0 / nDCG@5=0.7289 / primary=0.8645`. Judge your progress against the
**0.8645** ceiling, not against 1.0.

## Official baseline (what you need to beat, numbers on test)

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, self-check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |
| oracle (theoretical ceiling) | 1.0000 | 0.7289 | 0.8645 |

FM variance across 5 seeds: std(primary) ≈ 0.0008 on test. **Do not accept a candidate as an
"improvement" based on a single seed** — a difference smaller than ~0.002 is indistinguishable
from noise.

**The FM baseline's actual code is seeded as the first node in your search history (not just
this table of numbers).** When choosing what to work on, prefer making targeted, incremental
edits to that baseline node over rewriting a solution from scratch each time — the goal is to
improve it directly, the way a person would edit existing working code, not to reinvent an
unrelated approach on every attempt.

## Already checked by the organizers — do NOT spend iterations on this

1. **More static features** — adding 8 extra categorical domains (music_id, video_type,
   upload_type + 6 user-side buckets) on top of the base 5 (`user_id, video_id, author_id, tab,
   dur_bucket`) had no effect: 0.5940 vs 0.5950 primary, within noise.
2. **More FM capacity** — embedding dim k=8/16/32 gave 0.5895/0.5902/0.5887, barely moves.

Reason: the `user_id × video_id` cross already captures almost all the learnable signal in this
model; purely user-side first-order features don't change the ranking within a user at all
(proven separately: `item_pop × user_bias` gives a bit-for-bit identical result to plain
`item_pop`).

## Prioritized directions

⚠️ **Direction 1 (loss function) is already empirically exhausted — do NOT repeat it.** Over
6 independent runs with ranking loss (LambdaRank/LGBMRanker, multi-seed ensembling), test
primary consistently lands in a narrow **0.591–0.595** range (baseline 0.5946), and further
tuning of search hyperparameters (patience, num_drafts) does not break through this range.
The switch from pointwise to pairwise/listwise loss has already been made and is already
reflected in your predecessors' features.

**In THIS run, you must try direction 2 (user behavioral history) — it is completely
untouched, none of the previous candidates used interaction sequences.** A concrete,
moderately complex starting point (you don't need to build a full DIN/SIM with attention on
the first attempt — start simpler and add complexity as you succeed):

a. For each `user_id`, build their chronological interaction history from `train.csv`
   (+ `valid.csv`, when computing features for `test.csv` — both precede test in time,
   `date`/`time_ms` are already available for sorting).
b. Take the last K (e.g. 20-50) videos the user interacted with BEFORE the current row, embed
   their `video_id`/`author_id`/`tag`.
c. Simple start: mean-pool the history embeddings into one "recent interest vector" for the
   user, concatenate with existing features (user/video/author statistics) — this already
   counts as a full attempt at direction 2, attention is not required.
d. If that works and budget remains — replace mean-pooling with attention between the current
   candidate (target video embedding) and the history (DIN-style: weight each historical item
   by its similarity to the target) — stronger, but harder to implement correctly.
e. Don't forget multi-seed ensembling (see the requirement below) even for this direction.

Direction 1 (loss function) and direction 2 (history) are not mutually exclusive — the best
result will likely come from combining ranking loss (already proven) with sequence features
(new). Do not fall back to a pure static-tabular approach without history — that has already
been tried 6 times and repeating it will not produce new signal.

3. **Multi-task.** `train.csv`/`valid.csv` have `is_click, is_like, is_follow, is_comment,
   is_forward, play_time_ms` available — usable as auxiliary tasks for the main `long_view`
   objective. **They are absent from `test.csv`** — don't design an architecture that needs
   these columns at inference time.
4. **Watch-time modeling.** CWM's idea (`hyz20/CWM`): real watch time is censored (a video can
   end before the user would have stopped watching), so a one-sided/censored-regression loss
   is needed, not MSE. Implement the idea yourself on your chosen stack — don't pull in a
   dependency on the CWM repo itself (it uses `torch==1.6.0`, an old version, and a different
   target metric `long_view2`, incompatible with `evaluate.py`).
5. **Model swap** (DeepFM/DCN/xDeepFM etc.) — low priority: capacity has already been proven
   not to be the bottleneck (see above), the gain is more likely from loss/features/sequences.
6. **Temporal features and drift.** `hourmin`, `date`, the distribution difference between
   train and test.
7. **`log_random_valid.csv` as unbiased validation** — a check for whether the model has
   overfit to the biased (non-random) traffic of `train.csv`/`valid.csv`.

## Resource policy

This is a hackathon, external resources are open by default: any open-source library
(PyTorch, RecBole, TorchRec, LightGBM, …), any papers/docs/public solutions, any pretrained
weights. The note in the parent README about "numpy only" applies to the official baseline
script, not to your solution — use whatever you want.

The one hard constraint: **never compute the metric or peek at `test.csv` labels directly** —
they are physically absent from data_dir, but don't try to reconstruct them or search the
internet for this specific task's hidden test set. Reading papers/other people's solutions
for methods — allowed and encouraged; searching for a ready-made answer to this specific
run — not allowed.

### Method sources (a starting point, not a ceiling — search wider if needed)

- RecBole (`RUCAIBox/RecBole`, recbole.io) — a catalog of ~100 models by category
  (sequential/context-aware/knowledge-based), with descriptions and paper links.
- Microsoft Recommenders (`recommenders-team/recommenders`) — an algorithm table with
  "when to use what" descriptions + notebooks.
- Awesome-RSPapers (`RUCAIBox/Awesome-RSPapers`) — a curated paper list on ranking loss,
  multi-task, sequential rec, debiasing.
- CWM (`hyz20/CWM`) and its paper — directly relevant work on censored watch-time regression
  on this same family of data (see direction 4 above and the warning there).
- The original KuaiRand paper — domain description and feature construction.

## Process: what to log on each iteration

Besides the code, record for each candidate:
1. **Hypothesis** — what you're changing and why (with a reference to the method's source, if
   applicable).
2. **Expected effect** — why this should move `primary`.
3. **Result** — numbers on `valid.csv` (GAUC/nDCG@5/primary), **averaged over at least 2 seeds**
   (3 preferred) before accepting a candidate as better than the previous best.
4. Accept a candidate as the new "best" only if the gain in primary is > 0.002 — otherwise it's
   noise, not an improvement.

⚠️ **HARD REQUIREMENT, not a recommendation: every candidate must train a MINIMUM of 3
different seeds inside ONE script and average the predictions (ensembling) before scoring and
submitting.** A single fixed seed is not enough — below is empirical data from this same run
showing why a single seed systematically doesn't work, even when it gives a good valid score:

| candidate | seeds | valid primary | **real test primary** | vs baseline (0.5946) |
|---|---|---|---|---|
| LambdaRank | **[42,123,2024], averaged** | 0.601 | **0.5951** | **+0.0005 (the only win)** |
| LightGBM Ranker | [42,123,2024], averaged | 0.599 | 0.5917 | -0.0029 |
| MLP | single seed (unfixed) | 0.601 | 0.44 (!) worse than random | -0.15 |
| LightGBM (pointwise) | single seed | 0.5947 | 0.5843 | -0.0103 |
| LightGBM Ranker | single seed | 0.5988 | 0.5902 | -0.0044 |

The pattern is unambiguous across 5 independent candidates: **both seed-averaged candidates
beat baseline; not a single single-seed candidate beat baseline**, regardless of how good its
valid score was. A single seed can look great on valid and collapse on test purely from luck
of initialization/batch shuffling — this is not a bug, it's variance, and an unaveraged
valid-score is an unreliable signal for the final decision.

Concrete pattern (copy this structure exactly, don't skip it):
```python
seeds = [42, 123, 2024]          # minimum 3
valid_preds_per_seed, test_preds_per_seed = [], []
for seed in seeds:
    model = train_model(..., random_state=seed)   # or torch.manual_seed(seed) before init
    valid_preds_per_seed.append(model.predict(X_valid))
    test_preds_per_seed.append(model.predict(X_test))
valid_scores = np.mean(valid_preds_per_seed, axis=0)   # average the PREDICTIONS, not the metrics
test_scores  = np.mean(test_preds_per_seed, axis=0)
primary = evaluate(...)['primary']   # compute the metric AFTER averaging predictions
```
A candidate without this pattern (a single seed, no averaging across multiple runs) is
considered an incomplete solution, even if the code runs and the valid score looks good.

## Final submission format

CSV with header `row_id,user_id,video_id,score`, one row per row of `test.csv`, `row_id` — an
integer starting at 0 with no gaps, `score` — any real number (only the relative order within
a user is used), no NaN/Inf. `row_id/user_id/video_id` must match `test.csv` row-for-row —
this is checked separately before scoring. Your final solution must include code that reads
`test.csv` and produces such a file — that is the designated final submission.

⚠️ **Explicitly cast `row_id`/`user_id`/`video_id` to `int` before `to_csv()`**:
`submission["row_id"] = submission["row_id"].astype(int)`. Verified empirically: one candidate
wrote `row_id` as `0.0,1.0,2.0,...` (pandas silently upcasts an int column to float if a NaN
ends up in it anywhere along the way, e.g. via a `merge` with an unmatched key) — such a file
**will not pass** the format check (strict `int(row_id)` parsing rejects `"0.0"`), even if the
predictions themselves are correct. Check `submission.dtypes` before saving.

⚠️ **Also save your raw prediction arrays, not just `submission.csv`**, right before writing
the submission:
```python
np.save("./working/valid_scores.npy", final_valid_scores)  # 1D array, same row order as valid.csv
np.save("./working/test_scores.npy", final_test_scores)    # 1D array, same row order as test.csv
```
This lets the harness combine your candidate with other independently-good candidates into an
ensemble automatically (this session's single best result was exactly that: a weighted blend of
three architecturally different models, each beating individually what none of them beat alone
by blending). Skipping this doesn't break your candidate's own score, but it excludes it from
ever being used in an ensemble.

## Robustness and autonomy

- On a candidate error (exception, timeout, invalid input) — log it, roll back to the last
  valid checkpoint, continue with the next hypothesis. Do not stop the entire run because of
  one failed candidate.
- Work autonomously: don't ask for confirmation on questions already settled in this document
  (library choices, available features, submission format, candidate-acceptance rule).
- Track the best checkpoint by `valid.csv` throughout the run — the final scoring uses the
  best-by-validation version, not the last one.
