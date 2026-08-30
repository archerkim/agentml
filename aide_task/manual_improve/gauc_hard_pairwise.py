"""
GAUC-targeted pairwise loss with within-user hard positive/negative mining.

Research basis: GAUC is a per-user-weighted AUC. Standard pointwise (BCE) or NDCG-lambda
(LambdaRank) losses don't directly target it. PDAOM (arXiv:2304.09176, "Enhancing
Personalized Ranking With Differentiable Group AUC Optimization") builds a pairwise loss
from HARD positive/negative pairs grouped by user_id specifically to move GAUC. Hard-BPR
(arXiv:2403.19276) flags a real risk with naive hardest-first mining: it can amplify false
negatives (items the user would have liked but never saw). This implementation:

  - Groups training rows by user_id per batch (not random row-level minibatches).
  - For each user with >=1 positive AND >=1 negative in the batch, forms up to MAX_PAIRS
    pairs between currently-hard positives (low predicted score) and currently-hard
    negatives (high predicted score).
  - Selects hard examples via SOFTMAX-WEIGHTED SAMPLING (not deterministic top-k) - this is
    the mitigation for the false-negative risk Hard-BPR raises: a single always-hardest
    negative gets resampled every epoch and can dominate/overfit to a possible false
    negative, whereas weighted sampling still favors hard examples but doesn't fixate.
  - Loss on each pair: standard BPR/logistic -log(sigmoid(score_pos - score_neg)).

Architecture is DELIBERATELY the same FM (same 5 fields, same interaction math) as
baseline_seed.py / run 18's ancestor - the only experimental variable here is the loss
function and batching scheme, isolated on purpose so any effect is attributable to it.
Reimplemented in PyTorch (not the project's hand-rolled numpy FM) because this custom
batched/grouped training loop with per-user sampling is far more reliably correct with
autograd than hand-deriving gradients for it.
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
MAX_PAIRS_PER_USER = 4
HARD_TEMPERATURE = 1.0  # softmax temperature for hard-example sampling; lower = harder-only


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
    utr = train["user_id"].to_numpy()
    Xva = encode_split(valid)
    yva = valid["long_view"].to_numpy(dtype=np.float32)
    uva = valid["user_id"].to_numpy()
    Xte = encode_split(test)
    dim = int(sum(field_dims))
    return (Xtr, ytr, utr), (Xva, yva, uva), Xte, dim, test


class FM(nn.Module):
    """Same math as the project's numpy FM (baseline.py / baseline_seed.py), reimplemented
    in torch so autograd can handle the custom loss below."""

    def __init__(self, dim, k=16):
        super().__init__()
        self.V = nn.Embedding(dim, k)
        self.W = nn.Embedding(dim, 1)
        self.b = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.V.weight, std=0.01)
        nn.init.zeros_(self.W.weight)

    def forward(self, X):
        E = self.V(X)  # (B, F, k)
        S = E.sum(1)  # (B, k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum(dim=(1, 2)))
        linear = self.W(X).squeeze(-1).sum(1)
        return self.b + linear + inter  # (B,) raw logit


def build_user_groups(users):
    """user_id -> row indices, for grouping a training epoch into per-user batches."""
    groups = {}
    for i, u in enumerate(users):
        groups.setdefault(u, []).append(i)
    return {u: np.array(idx) for u, idx in groups.items()}


def hard_pairwise_loss(scores, labels, user_ids_in_batch, rng):
    """scores, labels: 1D tensors for ALL rows in this batch (multiple users' rows
    concatenated). Returns mean BPR loss over hard-mined pairs across all users that
    have both a positive and a negative in this batch."""
    device = scores.device
    losses = []
    detached = scores.detach()
    # group row positions (within this batch) by user
    order = np.argsort(user_ids_in_batch, kind="stable")
    sorted_users = user_ids_in_batch[order]
    boundaries = np.searchsorted(sorted_users, np.unique(sorted_users))
    boundaries = np.append(boundaries, len(sorted_users))

    for gi in range(len(boundaries) - 1):
        rows = order[boundaries[gi]:boundaries[gi + 1]]
        lab = labels[rows]
        pos_mask = lab == 1
        neg_mask = lab == 0
        n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())
        if n_pos == 0 or n_neg == 0:
            continue
        pos_idx = rows[pos_mask.cpu().numpy().astype(bool)]
        neg_idx = rows[neg_mask.cpu().numpy().astype(bool)]
        n_pairs = min(MAX_PAIRS_PER_USER, n_pos, n_neg)

        # hard negatives: high current score -> sample with prob proportional to
        # softmax(score / T) (favors hard, but not deterministic - false-negative mitigation)
        neg_scores = detached[neg_idx].cpu().numpy() / HARD_TEMPERATURE
        neg_probs = np.exp(neg_scores - neg_scores.max())
        neg_probs /= neg_probs.sum()
        sampled_neg = rng.choice(neg_idx, size=n_pairs, replace=len(neg_idx) < n_pairs, p=neg_probs)

        # hard positives: LOW current score -> sample with prob proportional to
        # softmax(-score / T)
        pos_scores = -detached[pos_idx].cpu().numpy() / HARD_TEMPERATURE
        pos_probs = np.exp(pos_scores - pos_scores.max())
        pos_probs /= pos_probs.sum()
        sampled_pos = rng.choice(pos_idx, size=n_pairs, replace=len(pos_idx) < n_pairs, p=pos_probs)

        s_pos = scores[sampled_pos]
        s_neg = scores[sampled_neg]
        pair_loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-9)
        losses.append(pair_loss)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.cat(losses).mean()


def train_one_seed(seed, dim, Xtr, ytr, utr, Xva_t, device, epochs=15, users_per_batch=256, lr=0.01, patience=3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = FM(dim, k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    user_groups = build_user_groups(utr)
    all_users = np.array(list(user_groups.keys()))

    Xtr_t = torch.tensor(Xtr, dtype=torch.long, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)

    best_state, best_primary, bad = None, -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(all_users)
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(all_users), users_per_batch):
            batch_users = all_users[start:start + users_per_batch]
            row_idx = np.concatenate([user_groups[u] for u in batch_users])
            batch_X = Xtr_t[row_idx]
            batch_y = ytr_t[row_idx]
            batch_users_arr = utr[row_idx]

            optimizer.zero_grad()
            scores = model(batch_X)
            loss = hard_pairwise_loss(scores, batch_y, batch_users_arr, rng)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * len(row_idx)
            n_batches += 1

        model.eval()
        with torch.no_grad():
            va_scores = model(Xva_t).cpu().numpy()
        va = evaluate(uva_global, yva_global, va_scores)
        print(f"  seed {seed} epoch {epoch:2d} | loss {total_loss/len(Xtr):.4f} | valid primary {va['primary']:.4f}")

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
    global uva_global, yva_global
    print("Loading data and encoding (same 5-field FM encoding as baseline)...")
    (Xtr, ytr, utr), (Xva, yva, uva), Xte, dim, test_df = load_and_encode()
    uva_global, yva_global = uva, yva

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Xva_t = torch.tensor(Xva, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.long, device=device)

    seeds = [42, 123, 2024]
    valid_preds_list, test_preds_list = [], []
    for seed in seeds:
        print(f"\n--- Training GAUC hard-pairwise model, seed {seed} ---")
        model = train_one_seed(seed, dim, Xtr, ytr, utr, Xva_t, device)
        model.eval()
        with torch.no_grad():
            valid_preds_list.append(model(Xva_t).cpu().numpy())
            test_preds_list.append(model(Xte_t).cpu().numpy())

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
