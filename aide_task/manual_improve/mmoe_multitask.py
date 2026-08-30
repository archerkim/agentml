"""
MMoE-style joint multi-task learning (research finding #4), done PROPERLY this time.

Earlier session attempt (run 7 improved, "click blend") trained a SEPARATE long_view model
and a SEPARATE click model, then blended their PREDICTIONS post-hoc - best blend weight came
out at 100% long_view / 0% click, i.e. no benefit. That's the wrong shape of "multi-task":
two independent models never share representations during training.

Kuaishou's own production approach (HoME/MMoE, arXiv:2408.05430) shares a set of expert
sub-networks across ALL tasks, with a per-task GATING network that learns how to combine
experts for that task. Gradients from EVERY task flow back into the shared experts during
training - so auxiliary tasks (is_click, is_like: much more abundant positives than
long_view) can shape a better shared representation, even though only the long_view TOWER's
output is used as the final ranking score (never blended with the auxiliary towers').

Architecture:
  shared embeddings (user, video, author, tab, dur_bucket) + GMF cross (proven to help)
    -> N_EXPERTS small MLPs (the "mixture of experts")
    -> per-task gate (softmax over experts, conditioned on the shared input)
    -> per-task weighted combination of expert outputs -> per-task tower -> per-task logit
  loss = long_view BCE (weight 1.0) + is_click BCE (weight 0.3) + is_like BCE (weight 0.15)
  final ranking score = ONLY the long_view tower's sigmoid output.

is_click/is_like are only used as TRAINING targets (train.csv/valid.csv have them); test.csv
correctly has no dependency on them since only the long_view tower's prediction path is used
at inference - the auxiliary towers exist purely to shape shared_repr during training.
"""
import sys

sys.path.insert(0, "./input")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from evaluate import evaluate

EMB_DIM = 16
N_EXPERTS = 4
EXPERT_HIDDEN = 32
TOWER_HIDDEN = 32
TASK_WEIGHTS = {"long_view": 1.0, "is_click": 0.3, "is_like": 0.15}


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

    def build_vocab(values):
        vocab = {}
        for v in values:
            if v not in vocab:
                vocab[v] = len(vocab) + 1  # 0 reserved for unknown
        return vocab

    user_vocab = build_vocab(train["user_id"].tolist())
    video_vocab = build_vocab(train["video_id"].tolist())
    author_vocab = build_vocab(train["author_id"].fillna(-1).astype(int).tolist())
    tab_vocab = build_vocab(train["tab"].tolist())
    edges = np.quantile(train["duration_ms"].to_numpy(), np.linspace(0, 1, 11)[1:-1])

    def encode_split(df):
        user_idx = np.array([user_vocab.get(u, 0) for u in df["user_id"]], dtype=np.int64)
        video_idx = np.array([video_vocab.get(v, 0) for v in df["video_id"]], dtype=np.int64)
        author_idx = np.array(
            [author_vocab.get(a, 0) for a in df["author_id"].fillna(-1).astype(int)], dtype=np.int64
        )
        tab_idx = np.array([tab_vocab.get(t, 0) for t in df["tab"]], dtype=np.int64)
        dur_bucket = np.searchsorted(edges, df["duration_ms"].to_numpy()).astype(np.int64)
        return {"user": user_idx, "video": video_idx, "author": author_idx, "tab": tab_idx, "dur_bucket": dur_bucket}

    Xtr = encode_split(train)
    Xva = encode_split(valid)
    Xte = encode_split(test)

    dims = {
        "user": len(user_vocab) + 1, "video": len(video_vocab) + 1,
        "author": len(author_vocab) + 1, "tab": len(tab_vocab) + 1, "dur_bucket": 11,
    }
    ytr = {t: train[t].to_numpy(dtype=np.float32) for t in TASK_WEIGHTS}
    yva_longview = valid["long_view"].to_numpy(dtype=np.float32)
    uva = valid["user_id"].to_numpy()
    return Xtr, ytr, Xva, yva_longview, uva, Xte, dims, test


class MMoE(nn.Module):
    def __init__(self, dims, tasks, emb_dim=EMB_DIM, n_experts=N_EXPERTS):
        super().__init__()
        self.user_emb = nn.Embedding(dims["user"], emb_dim, padding_idx=0)
        self.video_emb = nn.Embedding(dims["video"], emb_dim, padding_idx=0)
        self.author_emb = nn.Embedding(dims["author"], emb_dim, padding_idx=0)
        self.tab_emb = nn.Embedding(dims["tab"], emb_dim, padding_idx=0)
        self.dur_emb = nn.Embedding(dims["dur_bucket"], emb_dim)

        input_dim = emb_dim * 5 + emb_dim  # 5 base embeddings + GMF cross (user * video)

        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, EXPERT_HIDDEN), nn.ReLU())
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleDict({
            task: nn.Linear(input_dim, n_experts) for task in tasks
        })
        self.towers = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(EXPERT_HIDDEN, TOWER_HIDDEN), nn.ReLU(), nn.Linear(TOWER_HIDDEN, 1)
            )
            for task in tasks
        })
        self.tasks = tasks

    def shared_input(self, user, video, author, tab, dur_bucket):
        u_e = self.user_emb(user)
        v_e = self.video_emb(video)
        a_e = self.author_emb(author)
        t_e = self.tab_emb(tab)
        d_e = self.dur_emb(dur_bucket)
        gmf = u_e * v_e
        return torch.cat([u_e, v_e, a_e, t_e, d_e, gmf], dim=-1)

    def forward(self, user, video, author, tab, dur_bucket):
        x = self.shared_input(user, video, author, tab, dur_bucket)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, n_experts, hidden)
        logits = {}
        for task in self.tasks:
            gate_w = torch.softmax(self.gates[task](x), dim=-1).unsqueeze(-1)  # (B, n_experts, 1)
            combined = (gate_w * expert_outs).sum(dim=1)  # (B, hidden)
            logits[task] = self.towers[task](combined).squeeze(-1)  # (B,)
        return logits


def to_tensors(X, device):
    return {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in X.items()}


def train_one_seed(seed, dims, Xtr_t, ytr_t, Xva_t, device, epochs=15, bs=4096, lr=1e-3, patience=3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tasks = list(TASK_WEIGHTS.keys())
    model = MMoE(dims, tasks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    n = len(ytr_t["long_view"])
    best_state, best_primary, bad = None, -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            batch = {k: v[idx] for k, v in Xtr_t.items()}
            optimizer.zero_grad()
            logits = model(batch["user"], batch["video"], batch["author"], batch["tab"], batch["dur_bucket"])
            loss = sum(
                TASK_WEIGHTS[t] * criterion(logits[t], ytr_t[t][idx]) for t in tasks
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        model.eval()
        with torch.no_grad():
            va_logits = model(Xva_t["user"], Xva_t["video"], Xva_t["author"], Xva_t["tab"], Xva_t["dur_bucket"])
            va_scores = torch.sigmoid(va_logits["long_view"]).cpu().numpy()
        va = evaluate(uva_global, yva_global, va_scores)
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


def predict_longview(model, X_t):
    model.eval()
    with torch.no_grad():
        logits = model(X_t["user"], X_t["video"], X_t["author"], X_t["tab"], X_t["dur_bucket"])
        return torch.sigmoid(logits["long_view"]).cpu().numpy()


def run():
    global uva_global, yva_global
    print("Loading data and building shared-vocab encodings...")
    Xtr, ytr, Xva, yva_longview, uva, Xte, dims, test_df = load_and_encode()
    uva_global, yva_global = uva, yva_longview

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Xtr_t = to_tensors(Xtr, device)
    ytr_t = {t: torch.tensor(ytr[t], dtype=torch.float32, device=device) for t in TASK_WEIGHTS}
    Xva_t = to_tensors(Xva, device)
    Xte_t = to_tensors(Xte, device)

    seeds = [42, 123, 2024]
    valid_preds_list, test_preds_list = [], []
    for seed in seeds:
        print(f"\n--- Training MMoE multi-task model, seed {seed} ---")
        model = train_one_seed(seed, dims, Xtr_t, ytr_t, Xva_t, device)
        valid_preds_list.append(predict_longview(model, Xva_t))
        test_preds_list.append(predict_longview(model, Xte_t))

    final_valid_scores = np.mean(valid_preds_list, axis=0)
    final_test_scores = np.mean(test_preds_list, axis=0)

    final_valid = evaluate(uva, yva_longview, final_valid_scores)
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
