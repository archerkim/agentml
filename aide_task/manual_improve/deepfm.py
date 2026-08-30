"""
DeepFM (curriculum direction #5 - flagged low-priority in description.md, since capacity
has repeatedly not been the bottleneck this session, but trying it anyway per request).

FM component (linear + pairwise interaction, same math as baseline.py's FM) captures
low-order feature crosses; a deep MLP tower captures higher-order implicit interactions.
The defining DeepFM property (vs. generic Wide&Deep) is that BOTH components read from
the SAME embedding table - implemented here accordingly. Same 5-field base encoding as
baseline_seed.py (shared flat vocab via field offsets), isolating architecture as the one
experimental variable. Multi-seed ensembled, BCE loss (standard for DeepFM).
"""
import sys

sys.path.insert(0, "./input")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from evaluate import evaluate

FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]


def load_and_encode():
    train = pd.read_csv("./input/train.csv")
    valid = pd.read_csv("./input/valid.csv")
    test = pd.read_csv("./input/test.csv")
    video_feat = pd.read_csv("./input/video_features_basic_pure.csv")[["video_id", "author_id"]]

    for df in (train, valid, test):
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
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int64)

    def encode_split(df):
        r = raw(df)
        X = np.empty((len(df), len(FIELDS)), dtype=np.int64)
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


class DeepFM(nn.Module):
    def __init__(self, dim, n_fields, k=16, deep_hidden=(64, 32), dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(dim, k)  # SHARED between FM and deep components
        self.linear = nn.Embedding(dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

        layers = []
        prev = n_fields * k
        for h in deep_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, X):
        E = self.emb(X)  # (B, F, k) - shared embeddings
        S = E.sum(1)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum(dim=(1, 2)))
        fm_linear = self.linear(X).squeeze(-1).sum(1)
        fm_out = self.bias + fm_linear + fm_inter

        deep_out = self.deep(E.reshape(E.size(0), -1)).squeeze(-1)
        return fm_out + deep_out


def train_one_seed(seed, dim, n_fields, Xtr_t, ytr_t, Xva_t, uva, yva, device, epochs=20, bs=4096, lr=1e-3, patience=3):
    torch.manual_seed(seed)
    model = DeepFM(dim, n_fields).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    n = len(ytr_t)
    best_state, best_primary, bad = None, -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            logits = model(Xtr_t[idx])
            loss = criterion(logits, ytr_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        model.eval()
        with torch.no_grad():
            va_scores = torch.sigmoid(model(Xva_t)).cpu().numpy()
        va = evaluate(uva, yva, va_scores)
        print(f"  seed {seed} epoch {epoch:2d} | loss {total_loss/n:.4f} | valid primary {va['primary']:.4f}")

        if va["primary"] > best_primary + 1e-5:
            best_primary = va["primary"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  seed {seed} early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def run():
    print("Loading data and encoding (same 5-field FM encoding as baseline)...")
    (Xtr, ytr), (Xva, yva, uva), Xte, dim, test_df = load_and_encode()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Xtr_t = torch.tensor(Xtr, dtype=torch.long, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.long, device=device)

    seeds = [42, 123, 2024]
    valid_preds_list, test_preds_list = [], []
    for seed in seeds:
        print(f"\n--- Training DeepFM, seed {seed} ---")
        model = train_one_seed(seed, dim, len(FIELDS), Xtr_t, ytr_t, Xva_t, uva, yva, device)
        model.eval()
        with torch.no_grad():
            valid_preds_list.append(torch.sigmoid(model(Xva_t)).cpu().numpy())
            test_preds_list.append(torch.sigmoid(model(Xte_t)).cpu().numpy())

    final_valid_scores = np.mean(valid_preds_list, axis=0)
    final_test_scores = np.mean(test_preds_list, axis=0)

    final_valid = evaluate(uva, yva, final_valid_scores)
    print(
        f"\nEnsemble Validation Results -> GAUC: {final_valid['GAUC']:.4f}, "
        f"nDCG@5: {final_valid['nDCG@5']:.4f}, primary: {final_valid['primary']:.4f}"
    )

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({
        "row_id": test_df["row_id"].astype(int),
        "user_id": test_df["user_id"].astype(int),
        "video_id": test_df["video_id"].astype(int),
        "score": final_test_scores,
    })
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved ensemble submission to ./working/submission.csv successfully.")
    return final_valid


if __name__ == "__main__":
    run()
