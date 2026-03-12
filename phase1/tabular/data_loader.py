"""
Data loading and preprocessing for the Adult Income dataset.

The Adult Income (Census) dataset is a standard benchmark for XAI evaluation:
- Binary classification: income >50K (1) vs <=50K (0)
- 48,842 instances, 14 features (mixed numerical + categorical)
- Accessed via sklearn's fetch_openml (UCI ID 1590)

Preprocessing pipeline:
  Numerical  -> passthrough (XGBoost handles scaling internally)
  Categorical -> OrdinalEncoder (integers; tree models handle this natively)

Returns encoded arrays for the model AND a cleaned DataFrame for DiCE.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from phase1.tabular.config import (
    DATA_DIR, RANDOM_STATE, TEST_SIZE,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, FEATURE_NAMES, TARGET
)


def load_adult_income(force_reload: bool = False) -> dict:
    """
    Load and preprocess the Adult Income dataset.

    Returns
    -------
    dict with keys:
        X_train, X_test : np.ndarray  (encoded, for model/SHAP)
        y_train, y_test : np.ndarray  (0/1 labels)
        X_train_df, X_test_df : pd.DataFrame  (ordinal-encoded, for LIME/DiCE)
        feature_names : list[str]
        cat_indices : list[int]  indices of categorical features in X arrays
        num_indices : list[int]  indices of numerical features in X arrays
        preprocessor : fitted ColumnTransformer
        cat_encoder_categories : categories from OrdinalEncoder (for DiCE)
    """
    cache_path = DATA_DIR / "adult_processed.pkl"

    if cache_path.exists() and not force_reload:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Downloading Adult Income dataset from OpenML...")
    dataset = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = dataset.frame.copy()

    # Clean up column names and target
    df.columns = [c.strip().lower() for c in df.columns]
    df["class"] = (df["class"].str.strip().isin([">50K", ">50k"])).astype(int)

    # Drop rows with missing values (marked as '?' in original)
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)

    # Drop fnlwgt (sampling weight, not a meaningful predictor)
    if "fnlwgt" in df.columns:
        df = df.drop(columns=["fnlwgt"])

    # Verify all required features are present
    available = set(df.columns)
    for feat in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        if feat not in available:
            raise ValueError(f"Feature '{feat}' not found in dataset. Available: {available}")

    X_raw = df[FEATURE_NAMES].copy()
    y = df[TARGET].values.astype(int)

    # Build ColumnTransformer: numerical passthrough, categorical OrdinalEncoder
    # Output column order: NUMERICAL_FEATURES then CATEGORICAL_FEATURES (= FEATURE_NAMES)
    cat_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            ("cat", cat_encoder, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    X_encoded = preprocessor.fit_transform(X_raw).astype(np.float64)

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Also keep DataFrames for LIME and DiCE
    X_df = pd.DataFrame(X_encoded, columns=FEATURE_NAMES)
    X_train_df = X_df.iloc[: len(X_train)].reset_index(drop=True)
    X_test_df = X_df.iloc[len(X_train) :].reset_index(drop=True)

    num_indices = list(range(len(NUMERICAL_FEATURES)))
    cat_indices = list(range(len(NUMERICAL_FEATURES), len(FEATURE_NAMES)))

    # Compute baseline values (training mean) for faithfulness metrics
    baseline_values = X_train.mean(axis=0)

    # Compute per-feature std for stability perturbations
    feature_std = X_train.std(axis=0) + 1e-8  # avoid zero std

    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "feature_names": FEATURE_NAMES,
        "num_indices": num_indices,
        "cat_indices": cat_indices,
        "preprocessor": preprocessor,
        "baseline_values": baseline_values,
        "feature_std": feature_std,
        "df_full": df,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test instances.")
    print(f"Class balance (test): {y_test.mean():.3f} positive (>50K)")
    return result
