"""
DIN-style attention upgrade of the run-18 (baseline-seeded) winner.

What changed vs. the FM+sequence predecessor (logs/18-kuairand-pure-run1/best_solution.py):
  - That version fed K=10 history video_ids as separate FM fields, each with its OWN
    embedding vocab disjoint from the candidate video_id field - so "history item" and
    "candidate item" embeddings lived in unrelated spaces, and history was just summed
    into the FM interaction uniformly (no notion of "is this history item relevant to
    THIS candidate"). That's mean-pooling in disguise, not real attention.
  - Here: candidate video and history video_ids share ONE embedding table, so a
    similarity-based attention score (DIN: MLP over [hist, cand, hist-cand, hist*cand])
    is actually meaningful. Attention weights are masked on padding, softmaxed, and used
    to compute a candidate-conditioned "attended interest vector" instead of a flat mean.
  - Implemented in PyTorch (autograd) rather than hand-deriving attention backprop in the
    project's pure-numpy FM - safer and more reliable for a mechanism this new.
  - Keeps everything else that's already proven to matter this session: same 5 base FM
    fields, an FM-style GMF cross term (user_emb * video_emb), multi-seed ensembling
    (hard requirement), int-cast submission columns, evaluate.py as the only scorer.
"""
import sys

sys.path.insert(0, "./input")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from evaluate import evaluate

K = 10  # history length, matches the predecessor
EMB_DIM = 16


def build_history(train, valid):
    """Chronological per-user video_id history from train+valid (both precede test)."""
    all_logs = pd.concat([train, valid], ignore_index=True)
    all_logs = all_logs.sort_values(["user_id", "date", "time_ms"])
    history_map = {}
    for uid, group in all_logs.groupby("user_id"):
        vids = group["video_id"].tolist()
        padded = [0] * max(0, K - len(vids)) + vids[-K:]
        history_map[uid] = padded
    return history_map


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

    history_map = build_history(train, valid)

    def hist_matrix(df):
        return np.array([history_map.get(uid, [0] * K) for uid in df["user_id"]], dtype=np.int64)

    # Shared vocabs across ALL splits so candidate video_id and history video_id use the
    # SAME index space (this is the whole point - attention needs a shared embedding table).
    def build_vocab(values):
        vocab = {}
        for v in values:
            if v not in vocab:
                vocab[v] = len(vocab) + 1  # 0 reserved for padding/unknown
        return vocab

    user_vocab = build_vocab(train["user_id"].tolist())
    video_vocab = build_vocab(train["video_id"].tolist())
    author_vocab = build_vocab(train["author_id"].fillna(-1).astype(int).tolist())
    tab_vocab = build_vocab(train["tab"].tolist())

    edges = np.quantile(train["duration_ms"].to_numpy(), np.linspace(0, 1, 11)[1:-1])

    def encode_split(df):
        n = len(df)
        user_idx = np.array([user_vocab.get(u, 0) for u in df["user_id"]], dtype=np.int64)
        video_idx = np.array([video_vocab.get(v, 0) for v in df["video_id"]], dtype=np.int64)
        author_idx = np.array(
            [author_vocab.get(a, 0) for a in df["author_id"].fillna(-1).astype(int)], dtype=np.int64
        )
        tab_idx = np.array([tab_vocab.get(t, 0) for t in df["tab"]], dtype=np.int64)
        dur_bucket = np.searchsorted(edges, df["duration_ms"].to_numpy()).astype(np.int64)
        hist = hist_matrix(df)
        hist_idx = np.array(
            [[video_vocab.get(v, 0) for v in row] for row in hist], dtype=np.int64
        )
        return {
            "user": user_idx, "video": video_idx, "author": author_idx,
            "tab": tab_idx, "dur_bucket": dur_bucket, "hist": hist_idx,
        }

    Xtr = encode_split(train)
    ytr = train["long_view"].to_numpy(dtype=np.float32)
    Xva = encode_split(valid)
    yva = valid["long_view"].to_numpy(dtype=np.float32)
    uva = valid["user_id"].to_numpy()
    Xte = encode_split(test)

    dims = {
        "user": len(user_vocab) + 1, "video": len(video_vocab) + 1,
        "author": len(author_vocab) + 1, "tab": len(tab_vocab) + 1,
        "dur_bucket": 11,
    }
    return (Xtr, ytr), (Xva, yva, uva), Xte, dims, test


class DINAttentionFM(nn.Module):
    """FM-style base crosses + DIN attention over user history, candidate-conditioned."""

    def __init__(self, dims, emb_dim=EMB_DIM):
        super().__init__()
        self.user_emb = nn.Embedding(dims["user"], emb_dim, padding_idx=0)
        self.video_emb = nn.Embedding(dims["video"], emb_dim, padding_idx=0)  # SHARED: candidate + history
        self.author_emb = nn.Embedding(dims["author"], emb_dim, padding_idx=0)
        self.tab_emb = nn.Embedding(dims["tab"], emb_dim, padding_idx=0)
        self.dur_emb = nn.Embedding(dims["dur_bucket"], emb_dim)

        # DIN attention: score(hist_i, candidate) from [hist, cand, hist-cand, hist*cand]
        self.attn_mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        base_dim = emb_dim * 5  # user, video, author, tab, dur
        gmf_dim = emb_dim  # user * video cross (FM-style interaction)
        hist_dim = emb_dim * 2  # attended history + (attended * candidate) cross
        self.head = nn.Sequential(
            nn.Linear(base_dim + gmf_dim + hist_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user, video, author, tab, dur_bucket, hist):
        u_e = self.user_emb(user)
        v_e = self.video_emb(video)
        a_e = self.author_emb(author)
        t_e = self.tab_emb(tab)
        d_e = self.dur_emb(dur_bucket)

        hist_e = self.video_emb(hist)  # (B, K, emb) - SAME table as candidate video
        cand_expand = v_e.unsqueeze(1).expand(-1, hist_e.size(1), -1)  # (B, K, emb)
        attn_in = torch.cat(
            [hist_e, cand_expand, hist_e - cand_expand, hist_e * cand_expand], dim=-1
        )
        attn_score = self.attn_mlp(attn_in).squeeze(-1)  # (B, K)
        pad_mask = hist == 0  # padding/unknown history slots
        attn_score = attn_score.masked_fill(pad_mask, float("-inf"))
        # rows that are ALL padding (new user, no history) would softmax to NaN - guard them
        all_pad = pad_mask.all(dim=1)
        attn_score = torch.where(all_pad.unsqueeze(1), torch.zeros_like(attn_score), attn_score)
        attn_weight = torch.softmax(attn_score, dim=1).unsqueeze(-1)  # (B, K, 1)
        attended_hist = (attn_weight * hist_e).sum(dim=1)  # (B, emb) - candidate-conditioned interest
        attended_hist = torch.where(
            all_pad.unsqueeze(-1), torch.zeros_like(attended_hist), attended_hist
        )

        base = torch.cat([u_e, v_e, a_e, t_e, d_e], dim=-1)
        gmf = u_e * v_e
        hist_feats = torch.cat([attended_hist, attended_hist * v_e], dim=-1)

        x = torch.cat([base, gmf, hist_feats], dim=-1)
        return self.head(x).squeeze(-1)


def to_tensors(X, device):
    return {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in X.items()}


def train_one_seed(seed, dims, Xtr_t, ytr_t, Xva_t, device, epochs=15, bs=4096, lr=1e-3, patience=3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DINAttentionFM(dims).to(device)
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
            batch = {k: v[idx] for k, v in Xtr_t.items()}
            y = ytr_t[idx]
            optimizer.zero_grad()
            logits = model(batch["user"], batch["video"], batch["author"], batch["tab"], batch["dur_bucket"], batch["hist"])
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            va_logits = model(Xva_t["user"], Xva_t["video"], Xva_t["author"], Xva_t["tab"], Xva_t["dur_bucket"], Xva_t["hist"])
            va_scores = torch.sigmoid(va_logits).cpu().numpy()
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


def predict(model, X_t):
    model.eval()
    with torch.no_grad():
        logits = model(X_t["user"], X_t["video"], X_t["author"], X_t["tab"], X_t["dur_bucket"], X_t["hist"])
        return torch.sigmoid(logits).cpu().numpy()


def run():
    global uva_global, yva_global
    print("Loading data and building shared-vocab encodings...")
    (Xtr, ytr), (Xva, yva, uva), Xte, dims, test_df = load_and_encode()
    uva_global, yva_global = uva, yva

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Xtr_t = to_tensors(Xtr, device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = to_tensors(Xva, device)
    Xte_t = to_tensors(Xte, device)

    seeds = [42, 123, 2024]
    valid_preds_list, test_preds_list = [], []
    for seed in seeds:
        print(f"\n--- Training DIN-attention model, seed {seed} ---")
        model = train_one_seed(seed, dims, Xtr_t, ytr_t, Xva_t, device)
        valid_preds_list.append(predict(model, Xva_t))
        test_preds_list.append(predict(model, Xte_t))

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
