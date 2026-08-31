"""Combined FM solution: K=10 long_view history (shared video_id space) + K=3 click history (separate embedding block).
Merges best ideas from three candidates: target-aware shared-space interaction (C1), separate embedding architecture (C3),
and conservative K for secondary signal (C2). Multi-seed ensemble, causal construction, early stopping.
"""

import sys

sys.path.insert(0, "./input")

import os
import numpy as np
import pandas as pd
from evaluate import evaluate

FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
K_LV = 10  # long_view=1 history slots (shared video_id embedding → target-aware interaction)
K_CLK = 3  # is_click=1 history slots (separate embedding block → independent representation)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S**2).sum(1) - (E**2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
        ):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= (
                self.lr
                * (M / (1 - b1**self.t))
                / (np.sqrt(Vv / (1 - b2**self.t)) + eps)
            )
        self.b -= self.lr * g.sum()
        return float(
            -np.mean(
                y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)
            )
        )

    def predict(self, X, bs=200_000):
        return np.concatenate(
            [self.logits(X[i : i + bs])[0] for i in range(0, len(X), bs)]
        )


def load_and_encode():
    train = pd.read_csv("./input/train.csv")
    valid = pd.read_csv("./input/valid.csv")
    test = pd.read_csv("./input/test.csv")
    video_feat = pd.read_csv("./input/video_features_basic_pure.csv")[
        ["video_id", "author_id"]
    ]

    for name, df in (("train", train), ("valid", valid), ("test", test)):
        df.reset_index(drop=True, inplace=True)

    train = train.merge(video_feat, on="video_id", how="left")
    valid = valid.merge(video_feat, on="video_id", how="left")
    test = test.merge(video_feat, on="video_id", how="left")

    edges = np.quantile(train["duration_ms"].to_numpy(), np.linspace(0, 1, 11)[1:-1])

    def raw(df):
        dur_bucket = np.searchsorted(edges, df["duration_ms"].to_numpy()).astype(str)
        return [
            df["user_id"].astype(str).to_numpy(),
            df["video_id"].astype(str).to_numpy(),
            df["author_id"].astype(str).to_numpy(),
            df["tab"].astype(str).to_numpy(),
            dur_bucket,
        ]

    # --- Causal history construction (split-aware) ---
    def _user_hist(source_df, K, label_col):
        pos = source_df[source_df[label_col] == 1].copy()
        pos = pos.sort_values(["user_id", "date", "time_ms"])
        pos["rn"] = pos.groupby("user_id").cumcount(ascending=False)
        lk = pos[pos["rn"] < K].copy()
        lk["hp"] = K - 1 - lk["rn"]
        pv = lk.pivot_table(
            index="user_id", columns="hp", values="video_id", aggfunc="first"
        )
        pv.columns = [f"h{i}" for i in range(K)]
        return pv

    # Long_view positives (primary signal) — shared video_id embedding space
    hist_lv_tr = _user_hist(train, K_LV, "long_view")
    hist_lv_va = _user_hist(train, K_LV, "long_view")  # causal: train precedes valid
    hist_lv_te = _user_hist(
        pd.concat([train, valid], ignore_index=True), K_LV, "long_view"
    )  # causal: train+valid precedes test

    # Click positives (secondary signal) — separate embedding block
    hist_clk_tr = _user_hist(train, K_CLK, "is_click")
    hist_clk_va = _user_hist(train, K_CLK, "is_click")
    hist_clk_te = _user_hist(
        pd.concat([train, valid], ignore_index=True), K_CLK, "is_click"
    )

    def _hist_arr(df, ht, K):
        m = df[["user_id"]].merge(ht, left_on="user_id", right_index=True, how="left")
        return [m[f"h{i}"].fillna("__UNK__").astype(str).to_numpy() for i in range(K)]

    # --- Vocab and embedding layout ---
    train_raw = raw(train)
    vocabs = [dict() for _ in FIELDS]
    for i in range(len(FIELDS)):
        for v in train_raw[i]:
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]

    # Long_view history: SHARED video_id embedding space (zero extra params, target-aware interaction)
    lv_voc, lv_unk = vocabs[1], unk[1]

    # Click history: SEPARATE embedding block (independent representation, cross-space interactions)
    clk_voc = {v: i for i, v in enumerate(vocabs[1])}
    clk_unk = len(clk_voc)
    field_dims.append(clk_unk + 1)

    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    lv_off = int(offsets[1])  # video_id field offset (shared with long_view history)
    clk_off = int(offsets[-1])  # separate click-history block offset
    dim = int(sum(field_dims))
    N_TOT = len(FIELDS) + K_LV + K_CLK

    def encode_split(df, hist_lv, hist_clk):
        r = raw(df)
        X = np.empty((len(df), N_TOT), dtype=np.int32)
        for i in range(len(FIELDS)):
            X[:, i] = np.array([vocabs[i].get(v, unk[i]) for v in r[i]]) + offsets[i]
        # Long_view history → shared video_id space
        ha_lv = _hist_arr(df, hist_lv, K_LV)
        for i in range(K_LV):
            X[:, len(FIELDS) + i] = (
                np.array([lv_voc.get(v, lv_unk) for v in ha_lv[i]]) + lv_off
            )
        # Click history → separate embedding block
        ha_clk = _hist_arr(df, hist_clk, K_CLK)
        for i in range(K_CLK):
            X[:, len(FIELDS) + K_LV + i] = (
                np.array([clk_voc.get(v, clk_unk) for v in ha_clk[i]]) + clk_off
            )
        return X

    Xtr = encode_split(train, hist_lv_tr, hist_clk_tr)
    ytr = train["long_view"].to_numpy(dtype=np.float32)
    Xva = encode_split(valid, hist_lv_va, hist_clk_va)
    yva = valid["long_view"].to_numpy(dtype=np.float32)
    uva = valid["user_id"].to_numpy()
    Xte = encode_split(test, hist_lv_te, hist_clk_te)
    return (Xtr, ytr), (Xva, yva, uva), Xte, dim, test


def predict(X, model):
    """Predict scores for encoded feature matrix X using a trained FM model.

    Args:
        X: np.ndarray of shape (n_samples, n_fields), int32 encoded features.
        model: Trained FM instance.
    Returns:
        np.ndarray of shape (n_samples,) with predicted scores (logits).
    """
    return model.predict(X)


def run_fm(k=16, lr=0.001, epochs=40, bs=8192, patience=4, verbose=True):
    (Xtr, ytr), (Xva, yva, uva), Xte, dim, test_df = load_and_encode()
    seeds = [42, 123, 2024]
    valid_preds_per_seed = []
    test_preds_per_seed = []
    for seed in seeds:
        m = FM(dim, k=k, lr=lr, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), bs):
                m.step(Xtr[idx[i : i + bs]], ytr[idx[i : i + bs]])
            va = evaluate(uva, yva, m.predict(Xva))
            if verbose:
                print(
                    f"  seed {seed} epoch {ep:2d} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f}"
                )
            if va["primary"] > best + 1e-5:
                best, bad = va["primary"], 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"  seed {seed} early stop at epoch {ep}")
                    break
        m.V, m.W, m.b = best_state
        valid_preds_per_seed.append(m.predict(Xva))
        test_preds_per_seed.append(m.predict(Xte))

    # Average predictions across seeds (not metrics)
    final_valid_scores = np.mean(valid_preds_per_seed, axis=0)
    final_test_scores = np.mean(test_preds_per_seed, axis=0)
    final_valid = evaluate(uva, yva, final_valid_scores)
    print(f"\n=== Ensemble Validation (3 seeds averaged) ===")
    print(
        f"GAUC: {final_valid['GAUC']:.4f}, nDCG@5: {final_valid['nDCG@5']:.4f}, primary: {final_valid['primary']:.4f}"
    )

    # Save artifacts
    os.makedirs("./working", exist_ok=True)
    np.save("./working/valid_scores.npy", final_valid_scores)
    np.save("./working/test_scores.npy", final_test_scores)

    submission = pd.DataFrame(
        {
            "row_id": test_df["row_id"].astype(int),
            "user_id": test_df["user_id"].astype(int),
            "video_id": test_df["video_id"].astype(int),
            "score": final_test_scores,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print(f"Saved submission to ./working/submission.csv ({len(submission)} rows)")
    print(f"Submission dtypes:\n{submission.dtypes}")
    return final_valid


if __name__ == "__main__":
    run_fm()
