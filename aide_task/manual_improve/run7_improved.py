import sys

sys.path.insert(0, "./input")

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from evaluate import evaluate


def predict(
    train_path="./input/train.csv",
    valid_path="./input/valid.csv",
    test_path="./input/test.csv",
):
    print("Loading data...")
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)

    user_feat = pd.read_csv("./input/user_features_pure.csv")
    video_feat = pd.read_csv("./input/video_features_basic_pure.csv")

    print(
        "Feature engineering with user interaction history and advanced statistics..."
    )

    ALPHA = 20.0  # Bayesian smoothing prior strength, same convention as the official baseline

    def smooth(sum_col, count_col, global_mean, alpha=ALPHA):
        return (sum_col + alpha * global_mean) / (count_col + alpha)

    g_long_view = train["long_view"].mean()
    g_click = train["is_click"].mean()
    g_like = train["is_like"].mean()

    # train rows joined with author_id (video_feat carries author_id, train doesn't)
    train_with_author = train.merge(
        video_feat[["video_id", "author_id"]], on="video_id", how="left"
    )

    # Compute item statistics from train (+ Bayesian smoothing on proportions)
    item_stats = (
        train.groupby("video_id")
        .agg(
            item_impressions=("long_view", "count"),
            item_long_view_sum=("long_view", "sum"),
            item_click_sum=("is_click", "sum"),
            item_like_sum=("is_like", "sum"),
            item_play_time_mean=("play_time_ms", "mean"),
        )
        .reset_index()
    )
    item_stats["item_long_view_mean"] = smooth(
        item_stats["item_long_view_sum"], item_stats["item_impressions"], g_long_view
    )
    item_stats["item_click_mean"] = smooth(
        item_stats["item_click_sum"], item_stats["item_impressions"], g_click
    )
    item_stats["item_like_mean"] = smooth(
        item_stats["item_like_sum"], item_stats["item_impressions"], g_like
    )

    # Compute user historical engagement statistics from train (+ smoothing)
    user_stats = (
        train.groupby("user_id")
        .agg(
            user_impressions=("long_view", "count"),
            user_long_view_sum=("long_view", "sum"),
            user_click_sum=("is_click", "sum"),
            user_like_sum=("is_like", "sum"),
            user_play_time_mean=("play_time_ms", "mean"),
        )
        .reset_index()
    )
    user_stats["user_long_view_mean"] = smooth(
        user_stats["user_long_view_sum"], user_stats["user_impressions"], g_long_view
    )
    user_stats["user_click_mean"] = smooth(
        user_stats["user_click_sum"], user_stats["user_impressions"], g_click
    )
    user_stats["user_like_mean"] = smooth(
        user_stats["user_like_sum"], user_stats["user_impressions"], g_like
    )

    # Author-level statistics (mirrors item_stats, grouped by author_id instead of video_id)
    author_stats = (
        train_with_author.groupby("author_id")
        .agg(
            author_impressions=("long_view", "count"),
            author_long_view_sum=("long_view", "sum"),
            author_click_sum=("is_click", "sum"),
        )
        .reset_index()
    )
    author_stats["author_long_view_mean"] = smooth(
        author_stats["author_long_view_sum"],
        author_stats["author_impressions"],
        g_long_view,
    )
    author_stats["author_click_mean"] = smooth(
        author_stats["author_click_sum"], author_stats["author_impressions"], g_click
    )

    def prepare_df(df):
        df = df.merge(user_feat, on="user_id", how="left")
        df = df.merge(video_feat, on="video_id", how="left")
        df = df.merge(item_stats, on="video_id", how="left")
        df = df.merge(user_stats, on="user_id", how="left")
        df = df.merge(author_stats, on="author_id", how="left")

        # Fill missing values safely (unseen entities fall back to global mean, 0 for counts)
        df["item_impressions"] = df["item_impressions"].fillna(0)
        df["item_long_view_mean"] = df["item_long_view_mean"].fillna(g_long_view)
        df["item_click_mean"] = df["item_click_mean"].fillna(g_click)
        df["item_like_mean"] = df["item_like_mean"].fillna(g_like)
        df["item_play_time_mean"] = df["item_play_time_mean"].fillna(
            train["play_time_ms"].mean()
        )

        df["user_impressions"] = df["user_impressions"].fillna(0)
        df["user_long_view_mean"] = df["user_long_view_mean"].fillna(g_long_view)
        df["user_click_mean"] = df["user_click_mean"].fillna(g_click)
        df["user_like_mean"] = df["user_like_mean"].fillna(g_like)
        df["user_play_time_mean"] = df["user_play_time_mean"].fillna(
            train["play_time_ms"].mean()
        )

        df["author_impressions"] = df["author_impressions"].fillna(0)
        df["author_long_view_mean"] = df["author_long_view_mean"].fillna(g_long_view)
        df["author_click_mean"] = df["author_click_mean"].fillna(g_click)

        if "hourmin" in df.columns:
            df["hour"] = df["hourmin"] // 100
        else:
            df["hour"] = 0

        df["duration_sec"] = df["duration_ms"] / 1000.0

        # Cross (interaction) features between user and item affinity
        df["cross_long_view"] = df["user_long_view_mean"] * df["item_long_view_mean"]
        df["cross_click"] = df["user_click_mean"] * df["item_click_mean"]

        return df

    train_df = prepare_df(train)
    valid_df = prepare_df(valid)
    test_df = prepare_df(test)

    # Sort by user_id to compute group counts correctly for lambdarank
    print("Sorting data by user_id for pairwise ranking...")
    train_df = train_df.sort_values(by="user_id").reset_index(drop=True)
    valid_df = valid_df.sort_values(by="user_id").reset_index(drop=True)

    train_groups = train_df.groupby("user_id", sort=False).size().values
    valid_groups = valid_df.groupby("user_id", sort=False).size().values

    features = [
        "duration_ms",
        "duration_sec",
        "is_rand",
        "item_impressions",
        "item_long_view_mean",
        "item_click_mean",
        "item_like_mean",
        "item_play_time_mean",
        "user_impressions",
        "user_long_view_mean",
        "user_click_mean",
        "user_like_mean",
        "user_play_time_mean",
        "author_impressions",
        "author_long_view_mean",
        "author_click_mean",
        "cross_long_view",
        "cross_click",
        "hour",
        "register_days",
    ]

    cat_cols = ["tab", "video_type", "upload_type", "music_type", "music_id", "tag"]
    for c in cat_cols:
        if c in train_df.columns:
            train_df[c] = train_df[c].astype("category")
            valid_df[c] = valid_df[c].astype("category")
            test_df[c] = test_df[c].astype("category")
            features.append(c)

    X_train = train_df[features]
    X_valid = valid_df[features]
    X_test = test_df[features]

    def train_ensemble(y_train, y_valid, objective_metric="ndcg"):
        """Train a 3-seed LightGBM Lambdarank ensemble for a given target column,
        return averaged (valid_preds, test_preds). Used for both the main long_view
        target and the auxiliary is_click target (multi-task blend, no leakage: only
        the auxiliary model's PREDICTIONS are used, is_click is never a feature)."""
        valid_preds_list, test_preds_list = [], []
        for seed in [42, 123, 2024]:
            params = {
                "objective": "lambdarank",
                "metric": objective_metric,
                "ndcg_eval_at": [5],
                "boosting_type": "gbdt",
                "learning_rate": 0.04,
                "num_leaves": 45,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "random_state": seed,
                "n_jobs": -1,
                "verbose": -1,
            }
            train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
            valid_data = lgb.Dataset(
                X_valid, label=y_valid, group=valid_groups, reference=train_data
            )
            model = lgb.train(
                params,
                train_data,
                num_boost_round=400,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
            )
            valid_preds_list.append(model.predict(X_valid, num_iteration=model.best_iteration))
            test_preds_list.append(model.predict(X_test, num_iteration=model.best_iteration))
        return np.mean(valid_preds_list, axis=0), np.mean(test_preds_list, axis=0)

    print("Training LightGBM Lambdarank model with multi-seed ensembling (long_view)...")
    lv_valid_preds, lv_test_preds = train_ensemble(
        train_df["long_view"], valid_df["long_view"]
    )

    print("Training auxiliary LightGBM Lambdarank model (is_click, multi-task blend)...")
    click_valid_preds, click_test_preds = train_ensemble(
        train_df["is_click"], valid_df["is_click"]
    )

    # Blend: rank-normalize each score to [0,1] within-user before combining, so the two
    # models' differing score scales don't distort the blend; tune weight on valid.
    def rank_normalize(scores, user_ids):
        s = pd.Series(scores)
        return s.groupby(user_ids).rank(pct=True).values

    lv_valid_rank = rank_normalize(lv_valid_preds, valid_df["user_id"].values)
    click_valid_rank = rank_normalize(click_valid_preds, valid_df["user_id"].values)
    lv_test_rank = rank_normalize(lv_test_preds, test_df["user_id"].values)
    click_test_rank = rank_normalize(click_test_preds, test_df["user_id"].values)

    best_w, best_primary = 1.0, -1.0
    for w in [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6]:
        blend = w * lv_valid_rank + (1 - w) * click_valid_rank
        r = evaluate(valid_df["user_id"].values, valid_df["long_view"].values, blend)
        p = r["primary"] if isinstance(r, dict) else r[2]
        print(f"  blend weight w={w:.2f} (long_view) -> valid primary={p:.4f}")
        if p > best_primary:
            best_primary, best_w = p, w

    print(f"Best blend weight: w={best_w:.2f} (long_view) / {1-best_w:.2f} (click)")
    valid_df["score"] = best_w * lv_valid_rank + (1 - best_w) * click_valid_rank
    final_test_preds = best_w * lv_test_rank + (1 - best_w) * click_test_rank

    eval_results = evaluate(
        valid_df["user_id"].values,
        valid_df["long_view"].values,
        valid_df["score"].values,
    )

    if isinstance(eval_results, dict):
        gauc = eval_results.get("GAUC", 0.0)
        ndcg = eval_results.get("nDCG@5", 0.0)
        primary = eval_results.get("primary", 0.0)
    else:
        gauc, ndcg, primary = eval_results

    print(
        f"Validation Results -> GAUC: {gauc:.4f}, nDCG@5: {ndcg:.4f}, primary: {primary:.4f}"
    )

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "row_id": test_df["row_id"].astype(int),
            "user_id": test_df["user_id"].astype(int),
            "video_id": test_df["video_id"].astype(int),
            "score": final_test_preds,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved submission to ./working/submission.csv successfully.")
    return submission


if __name__ == "__main__":
    predict()
