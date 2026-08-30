"""Official FM baseline (baseline.py), adapted to run against the masked
train.csv/valid.csv/test.csv layout used by this AIDE task, instead of the raw
multi-file logs data.py expects. Same FM class, same encode() logic (5 FIELDS,
quantile-bucketed duration, train-only vocab), same hyperparameters (k=16,
lr=0.001, seed=0) as the official baseline_scores.json recipe.
"""
import sys

sys.path.insert(0, "./input")

import os
import numpy as np
import pandas as pd
from evaluate import evaluate

FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def load_and_encode():
    train = pd.read_csv("./input/train.csv")
    valid = pd.read_csv("./input/valid.csv")
    test = pd.read_csv("./input/test.csv")
    video_feat = pd.read_csv("./input/video_features_basic_pure.csv")[["video_id", "author_id"]]

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

    train_raw = raw(train)
    vocabs = [dict() for _ in FIELDS]
    for i in range(len(FIELDS)):
        for v in train_raw[i]:
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    def encode_split(df):
        r = raw(df)
        X = np.empty((len(df), len(FIELDS)), dtype=np.int32)
        for i in range(len(FIELDS)):
            X[:, i] = np.array([vocabs[i].get(v, unk[i]) for v in r[i]]) + offsets[i]
        return X

    Xtr = encode_split(train)
    ytr = train["long_view"].to_numpy(dtype=np.float32)
    Xva = encode_split(valid)
    yva = valid["long_view"].to_numpy(dtype=np.float32)
    uva = valid["user_id"].to_numpy()
    Xte = encode_split(test)
    dim = int(sum(field_dims))
    return (Xtr, ytr), (Xva, yva, uva), Xte, dim, test


def run_fm(k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    (Xtr, ytr), (Xva, yva, uva), Xte, dim, test_df = load_and_encode()
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f}")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    final_valid = evaluate(uva, yva, m.predict(Xva))
    print(f"Validation Results -> GAUC: {final_valid['GAUC']:.4f}, nDCG@5: {final_valid['nDCG@5']:.4f}, primary: {final_valid['primary']:.4f}")

    test_scores = m.predict(Xte)
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({
        "row_id": test_df["row_id"].astype(int),
        "user_id": test_df["user_id"].astype(int),
        "video_id": test_df["video_id"].astype(int),
        "score": test_scores,
    })
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved submission to ./working/submission.csv successfully.")
    return final_valid


if __name__ == "__main__":
    run_fm()
