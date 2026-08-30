"""
Censored watch-time regression (curriculum direction #4, CWM-inspired, untried all session).

Core idea: play_time_ms is a CENSORED observation of "true" engagement.
  - If a user stopped watching before the video ended (play_time_ms < duration_ms), that's
    an EXACT observation - they chose to stop there, use ordinary squared error.
  - If a user watched to the end or beyond (play_time_ms >= duration_ms), we only know their
    true desired watch time is AT LEAST play_time_ms - they might have kept watching if the
    video were longer. This is right-censored: penalize the model only for predicting BELOW
    the observed value, never for predicting above it (we don't know the true ceiling).

This is implemented as a custom LightGBM objective (grad/hess), not a built-in loss - LightGBM
has no censored-regression objective out of the box. Predicted (log) watch time is used
directly as the ranking score for evaluate() - no threshold/binarization needed, since the
task only cares about relative order within a user.

Model selection (early stopping) uses evaluate()'s primary metric on valid.csv via a custom
feval, exactly like the other LightGBM candidates this session - NOT the regression loss
itself, since that's not what's actually being scored.
"""
import sys

sys.path.insert(0, "./input")

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from evaluate import evaluate


def load_and_prepare():
    train = pd.read_csv("./input/train.csv")
    valid = pd.read_csv("./input/valid.csv")
    test = pd.read_csv("./input/test.csv")
    user_feat = pd.read_csv("./input/user_features_pure.csv")
    video_feat = pd.read_csv("./input/video_features_basic_pure.csv")

    ALPHA = 20.0

    def smooth(sum_col, count_col, global_mean, alpha=ALPHA):
        return (sum_col + alpha * global_mean) / (count_col + alpha)

    g_long_view = train["long_view"].mean()
    g_click = train["is_click"].mean()

    train_with_author = train.merge(
        video_feat[["video_id", "author_id"]], on="video_id", how="left"
    )

    item_stats = (
        train.groupby("video_id")
        .agg(
            item_impressions=("long_view", "count"),
            item_long_view_sum=("long_view", "sum"),
            item_click_sum=("is_click", "sum"),
        )
        .reset_index()
    )
    item_stats["item_long_view_mean"] = smooth(item_stats["item_long_view_sum"], item_stats["item_impressions"], g_long_view)
    item_stats["item_click_mean"] = smooth(item_stats["item_click_sum"], item_stats["item_impressions"], g_click)

    user_stats = (
        train.groupby("user_id")
        .agg(
            user_impressions=("long_view", "count"),
            user_long_view_sum=("long_view", "sum"),
            user_click_sum=("is_click", "sum"),
        )
        .reset_index()
    )
    user_stats["user_long_view_mean"] = smooth(user_stats["user_long_view_sum"], user_stats["user_impressions"], g_long_view)
    user_stats["user_click_mean"] = smooth(user_stats["user_click_sum"], user_stats["user_impressions"], g_click)

    author_stats = (
        train_with_author.groupby("author_id")
        .agg(author_impressions=("long_view", "count"), author_long_view_sum=("long_view", "sum"))
        .reset_index()
    )
    author_stats["author_long_view_mean"] = smooth(author_stats["author_long_view_sum"], author_stats["author_impressions"], g_long_view)

    def prepare(df, is_train):
        df = df.merge(user_feat, on="user_id", how="left")
        df = df.merge(video_feat, on="video_id", how="left")
        df = df.merge(item_stats, on="video_id", how="left")
        df = df.merge(user_stats, on="user_id", how="left")
        df = df.merge(author_stats, on="author_id", how="left")

        df["item_impressions"] = df["item_impressions"].fillna(0)
        df["item_long_view_mean"] = df["item_long_view_mean"].fillna(g_long_view)
        df["item_click_mean"] = df["item_click_mean"].fillna(g_click)
        df["user_impressions"] = df["user_impressions"].fillna(0)
        df["user_long_view_mean"] = df["user_long_view_mean"].fillna(g_long_view)
        df["user_click_mean"] = df["user_click_mean"].fillna(g_click)
        df["author_impressions"] = df["author_impressions"].fillna(0)
        df["author_long_view_mean"] = df["author_long_view_mean"].fillna(g_long_view)

        df["hour"] = (df["hourmin"] // 100) if "hourmin" in df.columns else 0
        df["log_duration"] = np.log1p(df["duration_ms"])

        if is_train:
            # censoring target/indicator - ONLY defined where play_time_ms exists (train)
            df["log_play_time"] = np.log1p(df["play_time_ms"].clip(lower=0))
            df["censored"] = (df["play_time_ms"] >= df["duration_ms"]).astype(int)
        return df

    train_df = prepare(train, is_train=True)
    valid_df = prepare(valid, is_train=False)
    test_df = prepare(test, is_train=False)

    features = [
        "duration_ms", "log_duration", "is_rand",
        "item_impressions", "item_long_view_mean", "item_click_mean",
        "user_impressions", "user_long_view_mean", "user_click_mean",
        "author_impressions", "author_long_view_mean",
        "hour", "register_days",
    ]
    cat_cols = ["tab", "video_type", "upload_type", "music_type", "music_id", "tag"]
    for c in cat_cols:
        if c in train_df.columns:
            train_df[c] = train_df[c].astype("category")
            valid_df[c] = valid_df[c].astype("category")
            test_df[c] = test_df[c].astype("category")
            features.append(c)

    return train_df, valid_df, test_df, features


def make_censored_objective(censor_flags):
    """Tobit-style one-sided loss: full squared error when uncensored, one-sided
    (penalize only underestimation) when censored (video watched to completion+)."""
    def objective(preds, train_data):
        y = train_data.get_label()
        diff = preds - y
        active = (censor_flags == 0) | (diff < 0)
        grad = np.where(active, diff, 0.0)
        hess = np.where(active, 1.0, 1e-6)
        return grad, hess
    return objective


def make_feval(uva, yva):
    def feval(preds, valid_data):
        r = evaluate(uva, yva, preds)
        return "primary", r["primary"], True
    return feval


def run():
    print("Loading data and engineering features...")
    train_df, valid_df, test_df, features = load_and_prepare()

    X_train = train_df[features]
    y_train_watchtime = train_df["log_play_time"].to_numpy()
    censor_flags = train_df["censored"].to_numpy()

    X_valid = valid_df[features]
    # valid.csv DOES carry play_time_ms too (train/valid keep outcome columns, only test
    # strips them) - use the real value for the Dataset label even though early stopping
    # is actually driven by feval (evaluate() on long_view), not this regression label.
    y_valid_watchtime = np.log1p(valid_df["play_time_ms"].clip(lower=0)).to_numpy()
    y_valid_longview = valid_df["long_view"].to_numpy()
    uva = valid_df["user_id"].to_numpy()

    X_test = test_df[features]

    print(f"Censored (watched to completion+) fraction in train: {censor_flags.mean():.3f}")

    seeds = [42, 123, 2024]
    valid_preds_list, test_preds_list = [], []

    for seed in seeds:
        print(f"\n--- Training censored watch-time model, seed {seed} ---")
        params = {
            # lightgbm==4.7.0: custom objectives are passed as a callable directly in
            # params["objective"] - the separate fobj= kwarg to lgb.train() was removed.
            "objective": make_censored_objective(censor_flags),
            "metric": "None",  # disable LightGBM's default regression metric entirely -
                                # early stopping must track ONLY our feval (real evaluate()
                                # primary on long_view), not a generic l2 on watch-time
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 45,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        train_set = lgb.Dataset(X_train, label=y_train_watchtime)
        valid_set = lgb.Dataset(X_valid, label=y_valid_watchtime, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=400,
            valid_sets=[valid_set],
            feval=make_feval(uva, y_valid_longview),
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )

        valid_preds_list.append(model.predict(X_valid, num_iteration=model.best_iteration))
        test_preds_list.append(model.predict(X_test, num_iteration=model.best_iteration))
        print(f"  seed {seed} best_iteration={model.best_iteration}")

    final_valid_scores = np.mean(valid_preds_list, axis=0)
    final_test_scores = np.mean(test_preds_list, axis=0)

    final_valid = evaluate(uva, y_valid_longview, final_valid_scores)
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
