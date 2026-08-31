"""Score EVERY node of a run against the held-out test labels (organizer-side analysis).

Deliberately outside the search loop: description.md forbids the agent from seeing test
labels, and verify_candidate() inside the loop is restricted to valid.csv. This is run by
hand afterwards and feeds nothing back.

    python3 score_all_steps.py logs/12-camp-01-sequence-depth [--jobs 4]
"""
import argparse, json, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
LABELS = HERE.parent / "held_out_test" / "test_labels.csv"
sys.path.insert(0, str(DATA))
from evaluate import evaluate  # noqa: E402


def score_one(step, code, root, timeout):
    ws = root / f"step{step}"
    (ws / "input").mkdir(parents=True, exist_ok=True)
    (ws / "working").mkdir(parents=True, exist_ok=True)
    for item in DATA.iterdir():
        d = ws / "input" / item.name
        if item.is_file() and not d.exists():
            shutil.copy(item, d)
    (ws / "solution.py").write_text(code)
    try:
        r = subprocess.run([sys.executable, "solution.py"], cwd=ws,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return step, None, "timeout"
    if r.returncode != 0:
        return step, None, f"crashed ({r.stderr.strip().splitlines()[-1][:60] if r.stderr.strip() else 'exit ' + str(r.returncode)})"
    sub_p = ws / "working" / "submission.csv"
    if not sub_p.exists():
        return step, None, "no submission.csv"
    sub = pd.read_csv(sub_p)
    labels = pd.read_csv(LABELS)
    if len(sub) != len(labels):
        return step, None, f"row mismatch {len(sub)}"
    m = sub.merge(labels, on="row_id", suffixes=("_s", "_l"))
    if (m["user_id_s"] != m["user_id_l"]).any() or (m["video_id_s"] != m["video_id_l"]).any():
        return step, None, "misaligned"
    if not np.isfinite(m["score"]).all():
        return step, None, "NaN/Inf scores"
    return step, evaluate(m["user_id_s"].to_numpy(), m["long_view"].to_numpy(),
                          m["score"].to_numpy()), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    j = json.load(open(run_dir / "journal.json"))
    id2step = {n["id"]: n["step"] for n in j["nodes"]}
    root = HERE / "_step_eval" / run_dir.name
    root.mkdir(parents=True, exist_ok=True)

    tasks = [(n["step"], n["code"]) for n in j["nodes"]]
    print(f"scoring {len(tasks)} nodes from {run_dir.name} with {args.jobs} workers ...", flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for step, res, status in ex.map(lambda t: score_one(t[0], t[1], root, args.timeout), tasks):
            results[step] = (res, status)
            print(f"  step {step}: {status}"
                  + (f"  test primary {res['primary']:.4f}" if res else ""), flush=True)

    print("\n| step | parent | valid primary | test GAUC | test nDCG@5 | test primary | vs baseline |")
    print("|---|---|---|---|---|---|---|")
    for n in j["nodes"]:
        s = n["step"]
        pid = j["node2parent"].get(n["id"])
        par = id2step.get(pid) if pid is not None else "—"
        v = n["metric"]["value"]
        vs = f"{v:.4f}" if v is not None else "buggy"
        res, status = results[s]
        if res:
            print(f"| {s} | {par} | {vs} | {res['GAUC']:.4f} | {res['nDCG@5']:.4f} | "
                  f"**{res['primary']:.4f}** | {res['primary'] - 0.5946:+.4f} |")
        else:
            print(f"| {s} | {par} | {vs} | — | — | _{status}_ | — |")
    print("\nFM baseline on test (published): 0.5946")


if __name__ == "__main__":
    main()
