"""
Train and evaluate an XGBoost classifier on the Adult Income dataset.

XGBoost is chosen as the base model because:
1. Strong performance on tabular data (commonly used in real-world settings)
2. TreeSHAP provides exact SHAP values efficiently (O(TLD^2) complexity)
3. Representative of gradient-boosted tree models used at Scania and similar industries
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb

from phase1.tabular.config import SAVED_MODEL_DIR, XGBOOST_PARAMS, RANDOM_STATE


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                force_retrain: bool = False) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier and report evaluation metrics.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test   : test data for evaluation
    force_retrain    : ignore cached model and retrain

    Returns
    -------
    Fitted XGBClassifier
    """
    model_path = SAVED_MODEL_DIR / "xgboost_adult.pkl"

    if model_path.exists() and not force_retrain:
        print(f"Loading cached model from {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        _report_metrics(model, X_test, y_test)
        return model

    print("Training XGBoost classifier...")
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    _report_metrics(model, X_test, y_test)

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    return model


def _report_metrics(model: xgb.XGBClassifier,
                    X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    print("\n=== XGBoost Model Performance ===")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
