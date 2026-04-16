"""
Metrics Loader — reads pre-computed metrics_summary.csv files.

All CSVs use wide format: one row, columns like LIME_AOPC, GradSHAP_AOPC, etc.
"""
from pathlib import Path
from typing import Optional
import pandas as pd


def load_metrics_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load a metrics_summary.csv. Returns None if missing."""
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def get_method_metrics(df: Optional[pd.DataFrame], method_prefix: str) -> dict:
    """
    Extract metrics for a given method from a wide-format DataFrame.

    Columns are named like  LIME_AOPC, GradSHAP_RankCorrelation, etc.
    method_prefix should match the column prefix exactly (case-sensitive),
    e.g. "LIME", "GradSHAP", "TimeSHAP", "IG", "SHAP", "CF".

    Returns {metric_name: float} dropping NaN values.
    """
    if df is None or df.empty:
        return {}

    prefix = method_prefix + "_"
    result = {}
    for col in df.columns:
        if col.startswith(prefix):
            metric_name = col[len(prefix):]
            val = df.iloc[0][col]
            if pd.notna(val):
                try:
                    result[metric_name] = float(val)
                except (ValueError, TypeError):
                    result[metric_name] = val
    return result


def format_metrics_for_prompt(metrics: dict) -> str:
    """
    Format a flat metrics dict as a compact string for LLM prompts.

    Example: "AOPC=0.4945, RankCorrelation=0.8755, MeanTime_s=0.0154"
    """
    if not metrics:
        return "no metrics available"
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)
