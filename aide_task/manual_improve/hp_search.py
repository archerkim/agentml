import sys

sys.path.insert(0, "./input")

import numpy as np
import pandas as pd
import lightgbm as lgb
from evaluate import evaluate

print("Loading data...")
train = pd.read_csv("./input/train.csv")
valid = pd.read_csv("./input/valid.csv")
user_feat = pd.read_csv("./input/user_features_pure.csv")
video_feat = pd.read_csv("./input/video_features_basic_pure.csv")

ALPHA = 20.0


def smooth(sum_col, count_col, global_mean, alpha=ALPHA):
    return (sum_col + alpha * global_mean) / (count_col + alpha)


g_long_view = train["long_view"].mean()
g_click = train["is_click"].mean()
g_like = train["is_like"].mean()

train_with_author = train.merge(video_feat[["video_id", "author_id"]], on="video_id", how="left")

item_stats = train.groupby("video_id").agg(
    item_impressions=("long_view", "count"), item_long_view_sum=("long_view", "sum"),
    item_click_sum=("is_click", "sum"), item_like_sum=("is_like", "sum"),
    item_play_time_mean=("play_time_ms", "mean"),
).reset_index()
item_stats["item_long_view_mean"] = smooth(item_stats["item_long_view_sum"], item_stats["item_impressions"], g_long_view)
item_stats["item_click_mean"] = smooth(item_stats["item_click_sum"], item_stats["item_impressions"], g_click)
item_stats["item_like_mean"] = smooth(item_stats["item_like_sum"], item_stats["item_impressions"], g_like)

user_stats = train.groupby("user_id").agg(
    user_impressions=("long_view", "count"), user_long_view_sum=("long_view", "sum"),
    user_click_sum=("is_click", "sum"), user_like_sum=("is_like", "sum"),
    user_play_time_mean=("play_time_ms", "mean"),
).reset_index()
user_stats["user_long_view_mean"] = smooth(user_stats["user_long_view_sum"], user_stats["user_impressions"], g_long_view)
user_stats["user_click_mean"] = smooth(user_stats["user_click_sum"], user_stats["user_impressions"], g_click)
user_stats["user_like_mean"] = smooth(user_stats["user_like_sum"], user_stats["user_impressions"], g_like)

author_stats = train_with_author.groupby("author_id").agg(
    author_impressions=("long_view", "count"), author_long_view_sum=("long_view", "sum"),
    author_click_sum=("is_click", "sum"),
).reset_index()
author_stats["author_long_view_mean"] = smooth(author_stats["author_long_view_sum"], author_stats["author_impressions"], g_long_view)
author_stats["author_click_mean"] = smooth(author_stats["author_click_sum"], author_stats["author_impressions"], g_click)


def prepare_df(df):
    df = df.merge(user_feat, on="user_id", how="left")
    df = df.merge(video_feat, on="video_id", how="left")
    df = df.merge(item_stats, on="video_id", how="left")
    df = df.merge(user_stats, on="user_id", how="left")
    df = df.merge(author_stats, on="author_id", how="left")
    df["item_impressions"] = df["item_impressions"].fillna(0)
    df["item_long_view_mean"] = df["item_long_view_mean"].fillna(g_long_view)
    df["item_click_mean"] = df["item_click_mean"].fillna(g_click)
    df["item_like_mean"] = df["item_like_mean"].fillna(g_like)
    df["item_play_time_mean"] = df["item_play_time_mean"].fillna(train["play_time_ms"].mean())
    df["user_impressions"] = df["user_impressions"].fillna(0)
    df["user_long_view_mean"] = df["user_long_view_mean"].fillna(g_long_view)
    df["user_click_mean"] = df["user_click_mean"].fillna(g_click)
    df["user_like_mean"] = df["user_like_mean"].fillna(g_like)
    df["user_play_time_mean"] = df["user_play_time_mean"].fillna(train["play_time_ms"].mean())
    df["author_impressions"] = df["author_impressions"].fillna(0)
    df["author_long_view_mean"] = df["author_long_view_mean"].fillna(g_long_view)
    df["author_click_mean"] = df["author_click_mean"].fillna(g_click)
    df["hour"] = (df["hourmin"] // 100) if "hourmin" in df.columns else 0
    df["duration_sec"] = df["duration_ms"] / 1000.0
    df["cross_long_view"] = df["user_long_view_mean"] * df["item_long_view_mean"]
    df["cross_click"] = df["user_click_mean"] * df["item_click_mean"]
    return df


train_df = prepare_df(train)
valid_df = prepare_df(valid)
train_df = train_df.sort_values(by="user_id").reset_index(drop=True)
valid_df = valid_df.sort_values(by="user_id").reset_index(drop=True)
train_groups = train_df.groupby("user_id", sort=False).size().values
valid_groups = valid_df.groupby("user_id", sort=False).size().values

features = [
    "duration_ms", "duration_sec", "is_rand", "item_impressions", "item_long_view_mean",
    "item_click_mean", "item_like_mean", "item_play_time_mean", "user_impressions",
    "user_long_view_mean", "user_click_mean", "user_like_mean", "user_play_time_mean",
    "author_impressions", "author_long_view_mean", "author_click_mean",
    "cross_long_view", "cross_click", "hour", "register_days",
]
cat_cols = ["tab", "video_type", "upload_type", "music_type", "music_id", "tag"]
for c in cat_cols:
    if c in train_df.columns:
        train_df[c] = train_df[c].astype("category")
        valid_df[c] = valid_df[c].astype("category")
        features.append(c)

X_train, y_train = train_df[features], train_df["long_view"]
X_valid, y_valid = valid_df[features], valid_df["long_view"]

print("Grid search (single seed=42 per config, for speed)...")
configs = [
    dict(learning_rate=0.04, num_leaves=45, num_boost_round=400, feature_fraction=0.8, bagging_fraction=0.8, min_data_in_leaf=20),
    dict(learning_rate=0.03, num_leaves=63, num_boost_round=600, feature_fraction=0.8, bagging_fraction=0.8, min_data_in_leaf=20),
    dict(learning_rate=0.03, num_leaves=45, num_boost_round=600, feature_fraction=0.7, bagging_fraction=0.7, min_data_in_leaf=50),
    dict(learning_rate=0.05, num_leaves=31, num_boost_round=400, feature_fraction=0.9, bagging_fraction=0.9, min_data_in_leaf=20),
    dict(learning_rate=0.02, num_leaves=63, num_boost_round=900, feature_fraction=0.8, bagging_fraction=0.8, min_data_in_leaf=30),
    dict(learning_rate=0.04, num_leaves=90, num_boost_round=400, feature_fraction=0.7, bagging_fraction=0.8, min_data_in_leaf=50),
]

results = []
for cfg in configs:
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
        "boosting_type": "gbdt", "bagging_freq": 1, "random_state": 42,
        "n_jobs": -1, "verbose": -1,
        "learning_rate": cfg["learning_rate"], "num_leaves": cfg["num_leaves"],
        "feature_fraction": cfg["feature_fraction"], "bagging_fraction": cfg["bagging_fraction"],
        "min_data_in_leaf": cfg["min_data_in_leaf"],
    }
    train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
    valid_data = lgb.Dataset(X_valid, label=y_valid, group=valid_groups, reference=train_data)
    model = lgb.train(params, train_data, num_boost_round=cfg["num_boost_round"],
                       valid_sets=[valid_data], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = model.predict(X_valid, num_iteration=model.best_iteration)
    r = evaluate(valid_df["user_id"].values, valid_df["long_view"].values, preds)
    p = r["primary"] if isinstance(r, dict) else r[2]
    results.append((p, cfg))
    print(f"  {cfg} -> valid primary={p:.4f} (best_iter={model.best_iteration})")

results.sort(reverse=True, key=lambda x: x[0])
print("\nBest config:", results[0])
