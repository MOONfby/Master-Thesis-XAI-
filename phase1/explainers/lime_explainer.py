"""
LIME (Local Interpretable Model-Agnostic Explanations) wrapper.

LIME approximates the black-box model locally around each instance using a
linear surrogate model trained on perturbed neighborhood samples.

Reference: Ribeiro et al., "Why Should I Trust You?", KDD 2016.

Key design choices for tabular data:
- Uses LimeTabularExplainer with the ordinal-encoded training data
- Categorical features are specified by index so LIME discretizes them correctly
- Returns feature attributions as a numpy array aligned with FEATURE_NAMES order
"""
import time
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from phase1.config import (
    LIME_N_SAMPLES, LIME_N_FEATURES, FEATURE_NAMES, RANDOM_STATE
)


class LIMEExplainer:
    """
    Wrapper around lime.lime_tabular.LimeTabularExplainer.

    Attributes
    ----------
    explainer : LimeTabularExplainer
    model     : sklearn-compatible classifier with predict_proba
    n_features : total number of features
    """

    def __init__(self, model, X_train: np.ndarray, cat_indices: list[int],
                 feature_names: list[str] = None):
        """
        Parameters
        ----------
        model      : fitted model with predict_proba method
        X_train    : training data (encoded), shape (n_train, n_features)
        cat_indices: indices of categorical features
        feature_names : feature names list
        """
        self.model = model
        self.n_features = X_train.shape[1]
        self.feature_names = feature_names or FEATURE_NAMES

        self.explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            class_names=["<=50K", ">50K"],
            categorical_features=cat_indices,
            mode="classification",
            discretize_continuous=True,
            random_state=RANDOM_STATE,
        )

    def explain(self, instance: np.ndarray, n_samples: int = LIME_N_SAMPLES,
                n_features: int = LIME_N_FEATURES) -> np.ndarray:
        """
        Generate LIME explanation for a single instance.

        Parameters
        ----------
        instance  : 1D array, shape (n_features,)
        n_samples : neighborhood samples for surrogate fitting
        n_features: max features in explanation

        Returns
        -------
        attributions : np.ndarray, shape (n_features,)
            Feature importances aligned with self.feature_names order.
            Features not included in the explanation get attribution 0.
        """
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict_proba,
            num_samples=n_samples,
            num_features=n_features,
            top_labels=1,
        )
        # Map LIME's (feature_index, weight) pairs back to full feature vector
        attributions = np.zeros(self.n_features)
        label = list(exp.local_exp.keys())[0]
        for feat_idx, weight in exp.local_exp[label]:
            attributions[feat_idx] = weight
        return attributions

    def explain_batch(self, instances: np.ndarray,
                      n_samples: int = LIME_N_SAMPLES,
                      n_features: int = LIME_N_FEATURES) -> np.ndarray:
        """
        Generate LIME explanations for multiple instances.

        Returns
        -------
        attributions : np.ndarray, shape (n_instances, n_features)
        """
        attributions = np.zeros((len(instances), self.n_features))
        for i, inst in enumerate(instances):
            attributions[i] = self.explain(inst, n_samples, n_features)
        return attributions

    def get_local_fidelity(self, instance: np.ndarray,
                           n_samples: int = LIME_N_SAMPLES,
                           n_features: int = LIME_N_FEATURES) -> float:
        """
        Return the R² score of the LIME surrogate on its neighborhood.

        This is an intrinsic faithfulness measure: how well does the local
        linear model approximate the black-box within the neighborhood?
        Higher R² → the explanation faithfully represents local model behavior.
        """
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict_proba,
            num_samples=n_samples,
            num_features=n_features,
            top_labels=1,
        )
        return exp.score  # sklearn R² of surrogate on neighborhood

    def measure_runtime(self, instances: np.ndarray, n_samples: int = LIME_N_SAMPLES,
                        n_features: int = LIME_N_FEATURES) -> dict:
        """Measure mean and std explanation time over a set of instances."""
        times = []
        for inst in instances:
            t0 = time.perf_counter()
            self.explain(inst, n_samples, n_features)
            times.append(time.perf_counter() - t0)
        return {"mean_time": np.mean(times), "std_time": np.std(times),
                "total_time": np.sum(times), "n_instances": len(instances)}
