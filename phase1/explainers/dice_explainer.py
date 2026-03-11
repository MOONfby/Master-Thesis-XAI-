"""
DiCE (Diverse Counterfactual Explanations) wrapper.

Counterfactual explanations answer "What is the minimum change to the input
that would flip the model's prediction?" — providing actionable, contrastive
explanations aligned with GDPR's 'right to explanation' for automated decisions.

Unlike LIME/SHAP (feature attribution), counterfactuals show:
  - WHAT to change (which features)
  - HOW MUCH to change (by how much)
  - IN WHICH DIRECTION (increase/decrease)

Reference: Mothilal et al., "Explaining Machine Learning Classifiers through
Diverse Counterfactual Explanations", FAccT 2020.

Design notes:
- DiCE works directly with the raw (ordinal-encoded) DataFrame
- We use the 'random' method for speed; 'genetic' gives better quality CFs
- Continuous features: the ordinal-encoded numerical features
- Categorical features: ordinal-encoded integer values treated as continuous
  (acceptable for tree models; DiCE's genetic/kdtree handles this better)
"""
import time
import numpy as np
import pandas as pd

from phase1.config import (
    CF_NUM_CFS, CF_DESIRED_CLASS, CF_METHOD,
    FEATURE_NAMES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    TARGET, RANDOM_STATE
)


class DiCEExplainer:
    """
    Wrapper around dice_ml for generating diverse counterfactual explanations.

    Requires dice-ml >= 0.9
    """

    def __init__(self, model, X_train_df: pd.DataFrame, y_train: np.ndarray,
                 feature_names: list[str] = None,
                 method: str = CF_METHOD):
        """
        Parameters
        ----------
        model       : fitted model with predict_proba (sklearn-compatible)
        X_train_df  : training data as DataFrame (ordinal-encoded)
        y_train     : training labels
        feature_names : list of feature names
        method      : DiCE search method ('random', 'genetic', 'kdtree')
        """
        import dice_ml
        from dice_ml import Dice

        self.model = model
        self.feature_names = feature_names or FEATURE_NAMES
        self.method = method
        self.n_features = len(self.feature_names)

        # Build training DataFrame with target column
        train_df = X_train_df.copy()
        train_df[TARGET] = y_train

        # DiCE Data object
        # All features treated as continuous (ordinal encoding makes this valid for trees)
        dice_data = dice_ml.Data(
            dataframe=train_df,
            continuous_features=self.feature_names,
            outcome_name=TARGET,
        )

        # DiCE Model object wrapping the sklearn-compatible classifier
        dice_model = dice_ml.Model(model=model, backend="sklearn")

        self.dice = Dice(dice_data, dice_model, method=method)

    def generate(self, instance_df: pd.DataFrame,
                 num_cfs: int = CF_NUM_CFS,
                 desired_class: int = CF_DESIRED_CLASS) -> dict:
        """
        Generate counterfactual explanations for a single instance.

        Parameters
        ----------
        instance_df  : single-row DataFrame (ordinal-encoded, no target column)
        num_cfs      : number of diverse counterfactuals to generate
        desired_class: target class (1 = >50K)

        Returns
        -------
        dict with:
            'cf_df'    : pd.DataFrame of counterfactual instances (num_cfs, n_features)
            'validity' : fraction of CFs with correct predicted class
            'success'  : bool, True if at least one valid CF was found
        """
        try:
            cf_result = self.dice.generate_counterfactuals(
                query_instances=instance_df,
                total_CFs=num_cfs,
                desired_class=desired_class,
                verbose=False,
            )
            cf_df = cf_result.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                return {"cf_df": None, "validity": 0.0, "success": False}

            # Drop target column if present
            if TARGET in cf_df.columns:
                cf_df = cf_df.drop(columns=[TARGET])

            # Reorder columns to match feature_names
            cf_df = cf_df[self.feature_names]

            # Compute validity: check predicted class of each CF
            cf_array = cf_df.values.astype(np.float64)
            cf_preds = self.model.predict(cf_array)
            validity = float((cf_preds == desired_class).mean())

            return {"cf_df": cf_df, "validity": validity, "success": True}

        except Exception as e:
            return {"cf_df": None, "validity": 0.0, "success": False, "error": str(e)}

    def generate_batch(self, instances_df: pd.DataFrame,
                       num_cfs: int = CF_NUM_CFS,
                       desired_class: int = CF_DESIRED_CLASS) -> list[dict]:
        """
        Generate counterfactuals for multiple instances.

        Returns list of result dicts (one per instance).
        """
        results = []
        for i in range(len(instances_df)):
            inst = instances_df.iloc[[i]]
            results.append(self.generate(inst, num_cfs, desired_class))
        return results

    def measure_runtime(self, instances_df: pd.DataFrame,
                        num_cfs: int = CF_NUM_CFS) -> dict:
        """Measure mean/std time to generate counterfactuals per instance."""
        times = []
        for i in range(len(instances_df)):
            inst = instances_df.iloc[[i]]
            t0 = time.perf_counter()
            self.generate(inst, num_cfs)
            times.append(time.perf_counter() - t0)
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "total_time": np.sum(times),
            "n_instances": len(instances_df),
        }
