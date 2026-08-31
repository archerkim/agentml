#!/usr/bin/env python3
"""Run a CAMPAIGN of AIDE runs, each pointed at a different research direction.

Why a campaign instead of one longer run
----------------------------------------
Measured, not assumed. Runs 9 and 10 were seeded identically, both had the full
findings.jsonl injected, and both independently converged on the same idea - "FM
baseline + one history-derived categorical field" - scoring 0.6035 and 0.6036.
Extra single runs buy repetition, not coverage, because nothing distinguishes one
run's brief from another's.

Three things are varied per run here:
  1. a DIRECTIVE (directives/*.md) naming the direction to explore, appended to the
     task description, so each run is briefed differently;
  2. improve_mode - "diff" for incremental feature work, "rewrite" where the
     directive needs an architecture change that a SEARCH/REPLACE patch structurally
     cannot express (you cannot patch an FM into a DeepFM). Both modes still set a
     parent on every node, so no orphan drafts appear either way;
  3. each directive demands an explicit internal hyperparameter sweep, so a single
     run reports a table of settings rather than one point estimate.

Runs are sequential: one 27B model on two GPUs, so parallel runs would only split
the same compute and halve each run's throughput.

findings.jsonl accumulates across runs, so each successive run is told what its
predecessors found. That file is the campaign's memory.

    python3 campaign.py                     # all directives, default budgets
    python3 campaign.py --only 01 03        # just those directives
    python3 campaign.py --wall_clock_sec 1800
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DIRECTIVES = HERE / "directives"
RUN_LOGS = HERE / "run_logs"

# improve_mode per directive. "rewrite" wherever the direction is an architecture or
# objective change rather than an added feature - those cannot be reached by atomic
# patches from the FM baseline, which is exactly why runs 9/10 never left FM.
PLAN = [
    ("01-sequence-depth",            "diff"),
    ("02-ranking-loss-plus-history", "rewrite"),
    ("03-deepfm-architecture",       "rewrite"),
    ("04-multitask",                 "rewrite"),
    ("05-watchtime-censored",        "rewrite"),
    ("06-hyperparameter-sweep",      "rewrite"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only directives whose name starts with one of these prefixes")
    ap.add_argument("--wall_clock_sec", type=float, default=3600,
                    help="per-run wall-clock ceiling (default 1h; the campaign total is this x n_runs)")
    # Effect sizes here are ~0.001-0.002 per step against an FM seed std of 0.0008, so
    # the competition's epsilon=0.002 marks almost every genuine improvement as a
    # failure: runs 9 and 10 both plateau-stopped at step 5 with hours of budget left.
    # 0.0005 still sits below the noise floor but lets real progress accumulate.
    ap.add_argument("--epsilon", type=float, default=0.0005)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max_merges", type=int, default=2)
    ap.add_argument("--reasoning_effort", default="medium",
                    help="medium is ~3x faster per step than max for statistically identical "
                         "results here (run 9: 0.6035 in 11min; run 10: 0.6036 in 33min)")
    ap.add_argument("--seed_from", default=None,
                    help="code file to seed as node 0 instead of the plain FM baseline. "
                         "Pointing this at an earlier directive's best_solution.py makes "
                         "each directive build on accumulated work rather than restarting "
                         "from 0.6015 - and gives rewrite-mode runs a known-correct data "
                         "pipeline to keep, which is where directive 02's KeyError came from. "
                         "Choose the file by VALIDATION score only; picking it by test score "
                         "would leak the held-out set into the search.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    plan = PLAN
    if args.only:
        plan = [(n, m) for (n, m) in plan if any(n.startswith(p) for p in args.only)]
    if not plan:
        raise SystemExit("no directives selected")

    RUN_LOGS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    summary_path = RUN_LOGS / f"campaign-{stamp}.jsonl"

    print(f"campaign: {len(plan)} runs, {args.wall_clock_sec:.0f}s each "
          f"(max {len(plan) * args.wall_clock_sec / 3600:.1f}h), effort={args.reasoning_effort}")
    for name, mode in plan:
        print(f"  - {name:32s} improve_mode={mode}")
    if args.dry_run:
        return

    for i, (name, mode) in enumerate(plan, 1):
        directive = DIRECTIVES / f"{name}.md"
        if not directive.exists():
            print(f"!! missing directive {directive}, skipping")
            continue

        exp_name = f"camp-{name}"
        log_file = RUN_LOGS / f"campaign-{stamp}-{name}.log"
        print(f"\n{'=' * 78}\n[{i}/{len(plan)}] {name}  (improve_mode={mode})\n"
              f"log -> {log_file}\n{'=' * 78}", flush=True)

        cmd = [
            sys.executable, "-u", "run_with_early_stop.py",
            "--config", args.config,
            "--directive", str(directive),
            "--exp_name", exp_name,
            "--improve_mode", mode,
            "--epsilon", str(args.epsilon),
            "--patience", str(args.patience),
            "--max_merges", str(args.max_merges),
            "--wall_clock_sec", str(args.wall_clock_sec),
        ]
        if args.seed_from:
            cmd += ["--seed_baseline", args.seed_from]
        t0 = time.time()
        with open(log_file, "w") as fh:
            rc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode
        elapsed = time.time() - t0

        # A failing run must not abort the campaign - the whole point is coverage across
        # directions, and description.md's own robustness rule says one failed candidate
        # should never stop the wider search.
        best = None
        try:
            stop = json.loads((HERE / "logs" / _latest_logdir(exp_name) / "stop_reason.json").read_text())
            best = stop.get("best_valid_metric")
        except Exception:
            pass
        rec = {"directive": name, "improve_mode": mode, "exit_code": rc,
               "elapsed_sec": round(elapsed, 1), "best_valid_metric": best,
               "log": str(log_file)}
        with open(summary_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{i}/{len(plan)}] {name}: rc={rc} best_valid={best} in {elapsed / 60:.1f} min", flush=True)

    print(f"\ncampaign summary -> {summary_path}")
    for line in summary_path.read_text().splitlines():
        r = json.loads(line)
        print(f"  {r['directive']:32s} best_valid={r['best_valid_metric']} "
              f"({r['elapsed_sec'] / 60:.0f} min, rc={r['exit_code']})")
    print("\nNext: python3 ensemble_runs.py   # blend the best solution from each direction")


def _latest_logdir(exp_name: str) -> str:
    """AIDE prefixes its log dir with a run index, so find the newest one for this name."""
    candidates = [p for p in (HERE / "logs").iterdir() if p.name.endswith(f"-{exp_name}")]
    if not candidates:
        raise FileNotFoundError(exp_name)
    return max(candidates, key=lambda p: p.stat().st_mtime).name


if __name__ == "__main__":
    main()
