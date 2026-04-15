"""Classical ML model wrapper for extractive sentence classification.

This module keeps model training logic separate from data loading and feature
engineering. It supports a small set of explainable or lightweight classifiers
appropriate for CPU-friendly academic projects.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from src.dataset_loader import SentenceClassificationExample
from src.feature_pipeline import SentenceFeatureConfig, SentenceFeatureExtractor


@dataclass(frozen=True)
class ExtractiveModelConfig:
    """Configuration for the extractive sentence importance classifier."""

    model_type: str = "logistic_regression"
    random_state: int = 42
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    n_estimators: int = 200
    hidden_layer_sizes: tuple[int, ...] = (64, 32)

    def __post_init__(self) -> None:
        """Validate model settings."""

        allowed_model_types = {"logistic_regression", "linear_svm", "random_forest", "mlp"}
        if self.model_type not in allowed_model_types:
            raise ValueError(f"model_type must be one of {sorted(allowed_model_types)}.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if not self.hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes must contain at least one layer.")


def _require_sklearn():
    """Import scikit-learn lazily with a friendly error message."""

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import LinearSVC
    except ImportError as error:
        raise ImportError(
            "The 'scikit-learn' package is required for model training. "
            "Install it with 'pip install scikit-learn'."
        ) from error

    return {
        "LogisticRegression": LogisticRegression,
        "LinearSVC": LinearSVC,
        "RandomForestClassifier": RandomForestClassifier,
        "MLPClassifier": MLPClassifier,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
    }


class ExtractiveSentenceClassifier:
    """Train, save, load, and apply a classical extractive classifier."""

    def __init__(
        self,
        model_config: ExtractiveModelConfig | None = None,
        feature_config: SentenceFeatureConfig | None = None,
    ) -> None:
        """Initialize the classifier and its feature extractor."""

        self.model_config = model_config or ExtractiveModelConfig()
        self.feature_config = feature_config or SentenceFeatureConfig()
        self.feature_extractor = SentenceFeatureExtractor(self.feature_config)
        self.model = self._build_model()
        self.is_fitted = False

    def _build_model(self):
        """Instantiate the configured scikit-learn classifier."""

        sklearn_objects = _require_sklearn()

        if self.model_config.model_type == "logistic_regression":
            return sklearn_objects["LogisticRegression"](
                max_iter=self.model_config.max_iter,
                random_state=self.model_config.random_state,
                class_weight=self.model_config.class_weight,
            )
        if self.model_config.model_type == "linear_svm":
            return sklearn_objects["LinearSVC"](
                max_iter=self.model_config.max_iter,
                random_state=self.model_config.random_state,
                class_weight=self.model_config.class_weight,
            )
        if self.model_config.model_type == "random_forest":
            return sklearn_objects["RandomForestClassifier"](
                n_estimators=self.model_config.n_estimators,
                random_state=self.model_config.random_state,
                class_weight=self.model_config.class_weight,
            )
        return sklearn_objects["MLPClassifier"](
            hidden_layer_sizes=self.model_config.hidden_layer_sizes,
            max_iter=self.model_config.max_iter,
            random_state=self.model_config.random_state,
        )

    def fit(self, examples: Sequence[SentenceClassificationExample]) -> "ExtractiveSentenceClassifier":
        """Fit the feature extractor and classifier on sentence examples."""

        if not examples:
            raise ValueError("Training examples cannot be empty.")

        features = self.feature_extractor.fit_transform(examples)
        labels = self.feature_extractor.extract_labels(examples)
        self.model.fit(features, labels)
        self.is_fitted = True
        return self

    def predict(self, examples: Sequence[SentenceClassificationExample]) -> np.ndarray:
        """Predict binary sentence labels."""

        self._validate_fitted()
        features = self.feature_extractor.transform(examples)
        return self.model.predict(features)

    def predict_scores(self, examples: Sequence[SentenceClassificationExample]) -> np.ndarray:
        """Predict continuous importance scores for ranking sentences."""

        self._validate_fitted()
        features = self.feature_extractor.transform(examples)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)
            return probabilities[:, 1]

        if hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(features)
            raw_scores = np.asarray(raw_scores, dtype=float)
            if raw_scores.ndim > 1:
                raw_scores = raw_scores[:, 0]
            score_min = raw_scores.min(initial=0.0)
            score_max = raw_scores.max(initial=0.0)
            if score_max - score_min == 0.0:
                return np.zeros_like(raw_scores, dtype=float)
            return (raw_scores - score_min) / (score_max - score_min)

        return self.model.predict(features).astype(float)

    def evaluate(self, examples: Sequence[SentenceClassificationExample]) -> Dict[str, float]:
        """Evaluate the classifier on a labeled sentence dataset."""

        self._validate_fitted()
        if not examples:
            raise ValueError("Evaluation examples cannot be empty.")

        sklearn_objects = _require_sklearn()
        gold_labels = self.feature_extractor.extract_labels(examples)
        predicted_labels = self.predict(examples)
        return {
            "accuracy": float(sklearn_objects["accuracy_score"](gold_labels, predicted_labels)),
            "precision": float(
                sklearn_objects["precision_score"](gold_labels, predicted_labels, zero_division=0)
            ),
            "recall": float(
                sklearn_objects["recall_score"](gold_labels, predicted_labels, zero_division=0)
            ),
            "f1": float(sklearn_objects["f1_score"](gold_labels, predicted_labels, zero_division=0)),
        }

    def save(self, output_path: str | Path) -> None:
        """Serialize the trained model bundle to disk."""

        self._validate_fitted()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_config": self.model_config,
            "feature_config": self.feature_config,
            "feature_extractor": self.feature_extractor,
            "model": self.model,
        }
        with path.open("wb") as file_handle:
            pickle.dump(payload, file_handle)

    @classmethod
    def load(cls, model_path: str | Path) -> "ExtractiveSentenceClassifier":
        """Load a serialized model bundle from disk."""

        path = Path(model_path)
        with path.open("rb") as file_handle:
            payload = pickle.load(file_handle)

        instance = cls(
            model_config=payload["model_config"],
            feature_config=payload["feature_config"],
        )
        instance.feature_extractor = payload["feature_extractor"]
        instance.model = payload["model"]
        instance.is_fitted = True
        return instance

    def _validate_fitted(self) -> None:
        """Ensure the model has been trained before inference."""

        if not self.is_fitted:
            raise ValueError("The classifier must be fitted before prediction or evaluation.")


__all__ = [
    "ExtractiveModelConfig",
    "ExtractiveSentenceClassifier",
]
