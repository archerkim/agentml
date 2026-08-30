# Technical Report: House Sales Price Prediction

## Introduction
The objective of this project is to predict residential sales prices (`SalePrice`) evaluated using Root Mean Squared Error (RMSE) on the log-transformed target variable. This report details the empirical findings and technical decisions across two progressive design iterations.

## Preprocessing
- **Target Transformation:** The target variable `SalePrice` was transformed using `np.log1p` to stabilize variance and align with the evaluation metric.
- **Missing Value Imputation:** Numerical features missing values were imputed using their respective median values. Categorical features missing values were imputed with the string `"Missing"`.
- **Feature Engineering:** Domain-specific variables were engineered to capture property size and age characteristics:
  - `TotalSF`: Sum of basement, 1st floor, and 2nd floor square footages.
  - `TotalBath`: Weighted sum of full and half bathrooms (basement and above ground).
  - `HouseAge`: Difference between the year sold (`YrSold`) and year built (`YearBuilt`).
  - `RemodelAge`: Difference between the year sold and year remodeled (`YearRemodAdd`).
  - `TotalPorchSF`: Aggregate square footage of open porches, 3-season porches, enclosed porches, screen porches, and wood decks.
- **Encoding & Scaling:** 
  - Iteration 1 utilized `OrdinalEncoder` for categorical features across tree-based models.
  - Iteration 2 introduced differential encoding: ordinal encoding for tree models (`HistGradientBoostingRegressor`, `ExtraTreesRegressor`) and one-hot encoding with standard scaling (`StandardScaler`) for linear models (`Ridge`). Skewed numerical features ($skew > 0.75$) were log-transformed for linear stability.

## Modelling Methods
- **Iteration 1 (Single Model Baseline):** Implemented scikit-learn's `HistGradientBoostingRegressor` (`max_iter=600`, `learning_rate=0.03`, `max_leaf_nodes=31`) within a 5-fold cross-validation scheme to address environment limitations preventing LightGBM usage.
- **Iteration 2 (Multi-Model Ensemble):** Constructed a heterogeneous ensemble combining `HistGradientBoostingRegressor`, `ExtraTreesRegressor` (`n_estimators=200`), and `Ridge` regression (`alpha=15.0`). Out-of-fold (OOF) predictions were combined using constrained optimization (`scipy.optimize.minimize`) to compute optimal blending weights.

## Results Discussion
- **Iteration 1:** Achieved a baseline Validation Log RMSE of **0.13386** using solely the histogram-based gradient boosting regressor.
- **Iteration 2:** The introduction of the multi-model ensemble (HistGB + ExtraTrees + Ridge) with log-transformed skewed features and optimized blending weights improved generalization, lowering the Validation Log RMSE to **0.12750**.

| Iteration | Models Included | Preprocessing Highlights | Validation Log RMSE |
| :--- | :--- | :--- | :--- |
| 1 | HistGradientBoosting | Median Imputation, Ordinal Encoding | 0.13386 |
| 2 | HistGB + ExtraTrees + Ridge | Skew Log-Transform, One-Hot/Ordinal Encodings, OOF Blending | **0.12750** |

## Future Work
- Explore advanced hyperparameter tuning using Optuna for individual ensemble constituents.
- Incorporate robust neural network architectures or gradient boosting variants (e.g., XGBoost, CatBoost) via container environments with pre-compiled C++ runtime dependencies.
- Investigate feature selection algorithms to eliminate noisy covariates and reduce multicollinearity in linear models.