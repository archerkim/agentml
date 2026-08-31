# AgentML — KuaiRand-Pure ranking research

This repository contains a reproducible ranking benchmark and an agent-driven experiment harness for the [KuaiRand-Pure](https://kuairand.com/) impression-log dataset. The task is to rank `long_view` predictions **within each user's shown impressions**, not to retrieve from a video catalogue.

It has two complementary entry points:

- the root-level starter kit is a small, dependency-light reference implementation and evaluator;
- [`aide_task/`](aide_task/) is a guarded AIDE research workflow that starts from an evaluated FM baseline, proposes controlled changes, validates them, records findings, and stops on a measured plateau.

## Results snapshot

`primary` is the mean of GAUC and nDCG@5; higher is better.

| Model / selection rule | Validation primary | Held-out primary |
|---|---:|---:|
| Official FM baseline | 0.6015 | 0.5946 |
| Causal-history FM, selected from an early run | 0.6036 | 0.5966 |
| Best validation-selected causal-history FM | **0.6048** | 0.5978 |
| Best retained held-out candidate | 0.6043 | **0.5981** |

The held-out labels are an organizer-side measurement only. They are never exposed to the AIDE search loop or used to choose its next candidate.

## Quick start: reference baseline

### Prerequisites

- Python 3.9+
- `numpy`
- KuaiRand-Pure source data

Download and unpack the source data in the repository root:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
python3 -m pip install numpy
```

Run the official FM baseline:

```bash
python3 baseline.py --model fm
```

The baseline reads `./KuaiRand-Pure/data` by default. Use `--data_dir` when the data lives elsewhere.

## Benchmark contract

| Item | Definition |
|---|---|
| Target | Binary `long_view` |
| Ranking scope | Each user's impressions in the evaluation split |
| Splits | train: 20220408–20220421; valid: 20220422–20220428; test: 20220429–20220508 |
| Metrics | GAUC, nDCG@5, and `primary = mean(GAUC, nDCG@5)` |
| Direction | Maximize every metric |
| Official test baseline | GAUC 0.6610, nDCG@5 0.5282, primary 0.5946 |
| Noise estimate | FM primary standard deviation ≈ 0.0008 across five seeds |

`evaluate.py` is the canonical metric implementation. Do not substitute a locally-written GAUC or nDCG implementation when comparing experiments.

The evaluation metric is deliberately not normalized to a practical maximum of 1.0: users with no positive impressions always receive nDCG@5 = 0. The oracle primary is approximately 0.8645.

## Submission format

Every submission must contain exactly one row for each test impression:

```csv
row_id,user_id,video_id,score
0,0,7531,-3.34176
```

`row_id` is the key. Do not join or align predictions on `(user_id, video_id)`: that pair is not unique in this dataset.

```bash
# Produce an example FM submission, validate its schema, or score a labeled validation split.
python3 submit.py --make  --split test  submission.csv
python3 submit.py --check --split test  submission.csv
python3 submit.py --score --split valid submission.csv
```

Scores may be any finite real values; only their within-user order matters. `row_id`, `user_id`, and `video_id` must be written as integers.

## AIDE research workflow

The AIDE workflow is designed for reliable, incremental experimentation rather than unconstrained code generation.

### 1. Prepare a safe agent data directory

The raw source log includes test outcomes, so do not give it directly to an agent. The preparation script creates an agent-visible data directory with labels stripped from `test.csv`, while retaining labels separately for organizer-side evaluation.

```bash
cd aide_task
python3 prepare_data.py
```

This produces `aide_task/data/` with train/validation labels and a label-free test split. It intentionally excludes `video_features_statistic_pure.csv`, whose full-period aggregates would leak outcomes.

### 2. Configure the model backend

The checked-in configuration expects a local Ollama model behind a small OpenAI-compatible proxy:

```bash
cd aide_task
cp .env.example .env
# Keep OPENAI_BASE_URL=http://127.0.0.1:11435/v1 for the maximum-reasoning proxy.
# Ensure the configured Ollama model is available locally.
```

The project was run with `aide-qwen3.8:27b`, a 131k context window, and `reasoning_effort=max`. The supplied [`Modelfile.aide-qwen3.8`](aide_task/Modelfile.aide-qwen3.8) documents that alias. AIDE itself, pandas, and PyTorch are required for this workflow; use the project's AIDE environment or install compatible versions before running it.

### 3. Run a focused experiment

`run_kuairand.sh` activates the AIDE environment and starts the proxy if needed:

```bash
cd aide_task
./run_kuairand.sh \
  --config config.yaml \
  --directive directives/01-sequence-depth.md \
  --exp_name history-depth
```

To run a sequence of independently directed experiments:

```bash
cd aide_task
python3 campaign.py --only 01 03 --wall_clock_sec 3600
```

Run campaigns sequentially: the local language model and GPU are shared resources, so concurrent campaigns compete rather than increase useful throughput.

## How the search works

The primary control loop is [`aide_task/run_with_early_stop.py`](aide_task/run_with_early_stop.py).

1. It injects the task description, prior findings, causal lessons, research notes, and a per-run directive into the agent prompt.
2. It evaluates a concrete seed solution as node zero in the journal. With `num_drafts: 1`, the seeded node consumes the draft budget, so subsequent work is an improvement of the best working solution or a debug pass over a failed one.
3. It executes every candidate against `valid.csv`, parses the review robustly, and re-executes any new best in a separate workspace before trusting it.
4. It asks for a causal attribution of each completed experiment and appends durable lessons for future runs.
5. It stops on the first of: validation plateau, wall-clock ceiling, or configured step cap. At plateau it can attempt a lineage-aware merge; after a run it can create a rank-normalized ensemble when multiple compatible lineages exist.

The harness also includes practical recovery safeguards: fixed AIDE run-index behavior, bounded LLM request timeouts, resilient structured-review parsing, seed-node rescue, candidate verification, and report compatibility patches.

### Experiment conventions

- Optimize on `valid.csv`; treat a primary move below about `0.002` as indistinguishable from seed noise unless replicated.
- Train at least three seeds and average predictions—not per-seed metrics—before judging a candidate.
- Build time-respecting features: validation history may use train only; test history may use train plus validation only.
- Persist `working/valid_scores.npy`, `working/test_scores.npy`, and `working/submission.csv` for every viable candidate.
- Keep model code and data-pipeline changes separate when possible. Whole-file rewrites are reserved for architecture changes; targeted `SEARCH`/`REPLACE` edits are the default.
- Do not read or use `held_out_test/test_labels.csv` from within the search loop. It is for post-run evaluation only.

## Post-run evaluation and ensembles

Score a finished solution once, outside the search process:

```bash
cd aide_task
python3 score_on_test.py logs/<run>/best_solution.py
```

For analysis of all journal nodes:

```bash
python3 score_all_steps.py logs/<run> --jobs 4
```

To blend diverse completed runs, optimize blend weights on validation predictions only:

```bash
python3 ensemble_runs.py --runs logs/<run-a> logs/<run-b>
```

The utility rank-normalizes scores within a user before blending, because score scales from different model families are not directly comparable.

## Repository layout

```text
.
├── baseline.py                 # Official random, popularity, and FM baselines
├── data.py                     # Dataset loading and reference feature encoding
├── evaluate.py                 # Canonical GAUC / nDCG@5 scorer
├── submit.py                   # Submission creation, validation, and valid scoring
├── ablation_features.py        # Reference feature ablations
├── baseline_scores.json        # Published baseline scores and noise estimate
├── held_out_test/              # Organizer-side labels; never agent-visible
└── aide_task/
    ├── prepare_data.py         # Creates the masked agent data directory
    ├── config.yaml             # AIDE model/search/execution configuration
    ├── run_kuairand.sh         # Safe local launcher and proxy bootstrap
    ├── run_with_early_stop.py  # Search, validation, memory, and stop controller
    ├── campaign.py             # Sequential directive-driven campaign runner
    ├── directives/             # Focused research briefs
    ├── findings.jsonl          # Cross-run result memory
    ├── lessons.jsonl           # Cross-run causal lessons
    ├── ensemble_runs.py        # Validation-tuned cross-run blending
    ├── score_on_test.py        # Organizer-side final evaluation
    └── run_logs/               # Compact, versioned launcher/run logs
```

Large source datasets, copied workspaces, and evaluator outputs are ignored by Git. They are reproducible local artifacts; retain the compact logs and journal summaries needed to understand an experiment.

## Reproducibility checklist

Before reporting an improvement:

1. Confirm the script uses the unmodified `evaluate.py`.
2. Verify label-free `test.csv` is the only test input visible during search.
3. Average three or more seeded prediction vectors.
4. Re-run the candidate in a clean workspace and confirm its validation score.
5. Validate the final CSV with `submit.py --check`.
6. Record the hypothesis, configuration, validation metrics, and failure mode or causal conclusion.

## License and data

This repository packages experiment code and derived artifacts, not the KuaiRand-Pure source dataset. Download and use the dataset under its original terms.
