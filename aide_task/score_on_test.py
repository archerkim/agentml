"""Score a candidate solution against the HELD-OUT test labels.

This is the organizer-side final evaluation, deliberately kept OUTSIDE the AIDE
search loop: description.md forbids the agent from ever seeing test labels, and
run_with_early_stop.py's verify_candidate() is likewise restricted to valid.csv.
This script is run by hand, after a run has finished, and never feeds anything
back into the search.

    python3 score_on_test.py logs/10-kuairand-pure-run1/best_solution.py
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
LABELS = HERE.parent / "held_out_test" / "test_labels.csv"

sys.path.insert(0, str(DATA))
from evaluate import evaluate  # noqa: E402


def build_workspace(root: Path) -> Path:
    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "working").mkdir(parents=True, exist_ok=True)
    for item in DATA.iterdir():
        dest = root / "input" / item.name
        if not dest.exists() and item.is_file():
            shutil.copy(item, dest)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--reuse-submission", action="store_true",
                    help="score an existing working/submission.csv without re-running")
    args = ap.parse_args()

    sol = Path(args.solution).resolve()
    ws = Path(args.workspace).resolve() if args.workspace else HERE / "_test_eval" / sol.parent.name
    build_workspace(ws)

    if not args.reuse_submission:
        runfile = ws / "solution.py"
        shutil.copy(sol, runfile)
        print(f"running {sol} in {ws} ...", flush=True)
        r = subprocess.run([sys.executable, runfile.name], cwd=ws, timeout=args.timeout)
        if r.returncode != 0:
            raise SystemExit(f"solution exited {r.returncode}")

    sub = pd.read_csv(ws / "working" / "submission.csv")
    labels = pd.read_csv(LABELS)

    # Alignment is checked, not assumed: (user_id, video_id) is NOT unique in this
    # dataset (3.06% duplicate pairs), so row_id is the only valid key.
    if len(sub) != len(labels):
        raise SystemExit(f"row count mismatch: submission {len(sub)} vs labels {len(labels)}")
    merged = sub.merge(labels, on="row_id", suffixes=("_sub", "_lab"), validate="one_to_one")
    if len(merged) != len(labels):
        raise SystemExit("row_id join did not cover every label row")
    bad = (merged["user_id_sub"] != merged["user_id_lab"]) | (merged["video_id_sub"] != merged["video_id_lab"])
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} rows disagree on user_id/video_id - submission is misaligned")
    if not np.isfinite(merged["score"]).all():
        raise SystemExit("submission contains NaN/Inf scores")

    res = evaluate(merged["user_id_sub"].to_numpy(),
                   merged["long_view"].to_numpy(),
                   merged["score"].to_numpy())
    print()
    print(f"=== TEST-SET SCORE for {sol.name} ({sol.parent.name}) ===")
    print(f"  GAUC    : {res['GAUC']:.4f}")
    print(f"  nDCG@5  : {res['nDCG@5']:.4f}")
    print(f"  primary : {res['primary']:.4f}")
    print()
    print(f"  FM baseline (official, test) : 0.5946")
    print(f"  delta vs baseline            : {res['primary'] - 0.5946:+.4f}")


if __name__ == "__main__":
    main()
