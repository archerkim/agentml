#!/usr/bin/env python3
"""Blend the best solution from several runs into one submission.

The in-run ensemble step in run_with_early_stop.py can only combine candidates from
DIFFERENT lineages inside one journal - and with num_drafts=1 there is only ever one
lineage, so it skips ("only 1 distinct npy-compatible good node"). This script is the
cross-run counterpart: each campaign run explores a different direction, so their best
solutions ARE architecturally diverse, which is precisely the condition under which
blending helped before.

That condition matters. findings.jsonl records both outcomes: a rank-normalised blend of
three architecturally different models was the best result of the earlier session
(test 0.5986, vs 0.5981 for the best single model), while blending near-duplicates was
worthless - run 8's in-run ensemble scored exactly its best ingredient because the
weight search put 100% on one candidate.

Weights are grid-searched on valid.csv ONLY. Test labels are never read here; score the
result separately with score_on_test.py.

    python3 ensemble_runs.py                        # every camp-* run
    python3 ensemble_runs.py --runs logs/12-camp-01-sequence-depth logs/13-camp-03-deepfm-architecture
"""
from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
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


def run_solution(sol: Path, ws: Path, timeout: int):
    """Execute one run's best_solution.py and collect its raw score arrays."""
    build_workspace(ws)
    shutil.copy(sol, ws / "solution.py")
    print(f"  running {sol} ...", flush=True)
    r = subprocess.run([sys.executable, "solution.py"], cwd=ws,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"  !! exited {r.returncode}; skipping. stderr tail:\n{r.stderr[-400:]}")
        return None
    v, t = ws / "working" / "valid_scores.npy", ws / "working" / "test_scores.npy"
    if not (v.exists() and t.exists()):
        print("  !! solution did not save valid_scores.npy/test_scores.npy; skipping "
              "(it cannot take part in an ensemble without them)")
        return None
    return np.load(v), np.load(t)


def rank_norm(x, groups):
    """Rank within user. The metric only uses within-user order, so raw score scales
    from different model families are not comparable and must be normalised first."""
    return pd.Series(x).groupby(groups).rank(pct=True).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="log dirs to blend (default: every logs/*camp-* with a best_solution.py)")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="_ensemble")
    args = ap.parse_args()

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        run_dirs = sorted(p for p in (HERE / "logs").iterdir()
                          if "camp-" in p.name and (p / "best_solution.py").exists())
    if len(run_dirs) < 2:
        raise SystemExit(f"need >=2 runs to blend, found {len(run_dirs)}")

    valid = pd.read_csv(DATA / "valid.csv")
    test = pd.read_csv(DATA / "test.csv")
    uva, yva = valid["user_id"].to_numpy(), valid["long_view"].to_numpy()
    uta = test["user_id"].to_numpy()

    preds = {}
    for d in run_dirs:
        got = run_solution(d / "best_solution.py", HERE / args.out / d.name, args.timeout)
        if got is None:
            continue
        vs, ts = got
        if len(vs) != len(valid) or len(ts) != len(test):
            print(f"  !! {d.name}: score array length mismatch; skipping")
            continue
        r = evaluate(uva, yva, vs)
        print(f"  {d.name}: valid primary {r['primary']:.4f}")
        preds[d.name] = (rank_norm(vs, uva), rank_norm(ts, uta))

    if len(preds) < 2:
        raise SystemExit(f"only {len(preds)} usable solutions; nothing to blend")

    names = list(preds)
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    best_w, best_p = None, -1.0
    for combo in itertools.product(grid, repeat=len(names) - 1):
        tail = 1.0 - sum(combo)
        if tail < -1e-9 or tail > 1.0 + 1e-9:
            continue
        w = list(combo) + [max(0.0, tail)]
        blend = sum(wi * preds[n][0] for wi, n in zip(w, names))
        p = evaluate(uva, yva, blend)["primary"]
        if p > best_p:
            best_p, best_w = p, w

    print("\n=== best blend (weights tuned on valid.csv only) ===")
    for n, w in zip(names, best_w):
        print(f"  {w:5.2f}  {n}")
    fv = sum(wi * preds[n][0] for wi, n in zip(best_w, names))
    ft = sum(wi * preds[n][1] for wi, n in zip(best_w, names))
    r = evaluate(uva, yva, fv)
    print(f"  ensemble valid: GAUC {r['GAUC']:.4f}  nDCG@5 {r['nDCG@5']:.4f}  primary {r['primary']:.4f}")

    out = HERE / args.out
    (out / "working").mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({
        "row_id": test["row_id"].astype(int),
        "user_id": test["user_id"].astype(int),
        "video_id": test["video_id"].astype(int),
        "score": ft,
    })
    sub.to_csv(out / "working" / "submission.csv", index=False)
    np.save(out / "working" / "valid_scores.npy", fv)
    np.save(out / "working" / "test_scores.npy", ft)
    print(f"\nwrote {out / 'working' / 'submission.csv'}")
    print(f"score it with:  python3 score_on_test.py --reuse-submission "
          f"--workspace {out} <any-solution-path>")


if __name__ == "__main__":
    main()
