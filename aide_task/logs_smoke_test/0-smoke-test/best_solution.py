import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate target and IDs
train_id = train_df["Id"]
test_id = test_df["Id"]
y_train_log = np.log1p(train_df["SalePrice"])

# Drop target and Id from features
X_train = train_df.drop(columns=["Id", "SalePrice"])
X_test = test_df.drop(columns=["Id"])

# Combine train and test for consistent preprocessing
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# Domain feature engineering
df_all["TotalSF"] = (
    df_all["TotalBsmtSF"].fillna(0)
    + df_all["1stFlrSF"].fillna(0)
    + df_all["2ndFlrSF"].fillna(0)
)
df_all["TotalBath"] = (
    df_all["FullBath"].fillna(0)
    + 0.5 * df_all["HalfBath"].fillna(0)
    + df_all["BsmtFullBath"].fillna(0)
    + 0.5 * df_all["BsmtHalfBath"].fillna(0)
)
df_all["HouseAge"] = df_all["YrSold"] - df_all["YearBuilt"]
df_all["RemodelAge"] = df_all["YrSold"] - df_all["YearRemodAdd"]
df_all["TotalPorchSF"] = (
    df_all["OpenPorchSF"].fillna(0)
    + df_all["3SsnPorch"].fillna(0)
    + df_all["EnclosedPorch"].fillna(0)
    + df_all["ScreenPorch"].fillna(0)
    + df_all["WoodDeckSF"].fillna(0)
)

# Identify feature types
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()

# Fill missing numerical values and log transform skewed features
for col in num_cols:
    df_all[col] = df_all[col].fillna(df_all[col].median())
    if df_all[col].skew() > 0.75:
        df_all[col] = np.log1p(np.maximum(0, df_all[col]))

# Ordinal encoding for tree-based models
df_trees = df_all.copy()
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_trees[cat_cols] = encoder.fit_transform(
    df_trees[cat_cols].fillna("Missing").astype(str)
)

# One-hot encoding for linear models
df_linear = pd.get_dummies(df_all, columns=cat_cols, drop_first=True)

# Split datasets
X_train_tree = df_trees.iloc[: len(train_df)].copy()
X_test_tree = df_trees.iloc[len(train_df) :].copy()

X_train_lin = df_linear.iloc[: len(train_df)].copy()
X_test_lin = df_linear.iloc[len(train_df) :].copy()

# 5-Fold Cross Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_hgb = np.zeros(len(X_train))
oof_et = np.zeros(len(X_train))
oof_ridge = np.zeros(len(X_train))

test_hgb = np.zeros(len(X_test))
test_et = np.zeros(len(X_test))
test_ridge = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tree, y_train_log)):
    # Tree features train/val
    X_tr_t, y_tr = X_train_tree.iloc[train_idx], y_train_log.iloc[train_idx]
    X_va_t, y_va = X_train_tree.iloc[val_idx], y_train_log.iloc[val_idx]

    # Model 1: HistGradientBoosting
    hgb = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.03, max_leaf_nodes=31, random_state=42
    )
    hgb.fit(X_tr_t, y_tr)
    oof_hgb[val_idx] = hgb.predict(X_va_t)
    test_hgb += hgb.predict(X_test_tree) / kf.n_splits

    # Model 2: ExtraTreesRegressor
    et = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    et.fit(X_tr_t, y_tr)
    oof_et[val_idx] = et.predict(X_va_t)
    test_et += et.predict(X_test_tree) / kf.n_splits

    # Linear features train/val
    X_tr_l = X_train_lin.iloc[train_idx]
    X_va_l = X_train_lin.iloc[val_idx]

    scaler = StandardScaler()
    X_tr_l_scaled = scaler.fit_transform(X_tr_l)
    X_va_l_scaled = scaler.transform(X_va_l)
    X_test_l_scaled = scaler.transform(X_test_lin)

    # Model 3: Ridge
    ridge = Ridge(alpha=15.0, random_state=42)
    ridge.fit(X_tr_l_scaled, y_tr)
    oof_ridge[val_idx] = ridge.predict(X_va_l_scaled)
    test_ridge += ridge.predict(X_test_l_scaled) / kf.n_splits


# Optimize ensemble weights on OOF predictions
def objective(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_hgb + w2 * oof_et + w3 * oof_ridge
    return np.sqrt(mean_squared_error(y_train_log, pred))


res = minimize(
    objective,
    [0.5, 0.25, 0.25],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: 1 - sum(w)},
)
weights = res.x

oof_blend = weights[0] * oof_hgb + weights[1] * oof_et + weights[2] * oof_ridge
cv_rmse = np.sqrt(mean_squared_error(y_train_log, oof_blend))
print(f"Validation Log RMSE: {cv_rmse:.5f}")

# Prepare submission file
test_blend = weights[0] * test_hgb + weights[1] * test_et + weights[2] * test_ridge
sub = pd.DataFrame({"Id": test_id, "SalePrice": np.expm1(test_blend)})

os.makedirs("./working", exist_ok=True)
sub.to_csv("./working/submission.csv", index=False)
