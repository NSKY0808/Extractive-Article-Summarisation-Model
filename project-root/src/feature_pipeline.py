"""Feature engineering utilities for classical extractive summarization models.

This module combines sparse TF-IDF sentence representations with small,
interpretable numeric features such as sentence position and length. The output
is designed for classical binary classifiers like Logistic Regression, SVM, or
Random Forest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .data_pipeline import NAMED_ENTITY_PATTERN, word_tokenize
from .dataset_loader import SentenceClassificationExample


@dataclass(frozen=True)
class SentenceFeatureConfig:
    """Configuration for sentence-level feature extraction."""

    max_tfidf_features: int = 5000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)
    use_tfidf: bool = True

    def __post_init__(self) -> None:
        """Validate feature extractor settings."""

        if self.max_tfidf_features < 1:
            raise ValueError("max_tfidf_features must be at least 1.")
        if self.min_df < 1:
            raise ValueError("min_df must be at least 1.")
        if not 0.0 < self.max_df <= 1.0:
            raise ValueError("max_df must be between 0.0 and 1.0.")
        if self.ngram_range[0] < 1 or self.ngram_range[1] < self.ngram_range[0]:
            raise ValueError("ngram_range must be a valid positive interval.")


def count_named_entities(sentence: str) -> int:
    """Count simple named-entity hints using capitalized phrase matching."""

    return len(NAMED_ENTITY_PATTERN.findall(sentence))


def compute_dense_features(example: SentenceClassificationExample) -> List[float]:
    """Build small explainable numeric features for one sentence."""

    tokens = word_tokenize(example.sentence_text)
    token_count = len(tokens)
    sentence_count = max(example.sentence_count, 1)
    denominator = max(sentence_count - 1, 1)

    relative_position = example.sentence_index / denominator
    reverse_position = (sentence_count - example.sentence_index - 1) / denominator
    normalized_length = token_count / 50.0
    named_entity_count = float(count_named_entities(example.sentence_text))
    has_number = float(any(character.isdigit() for character in example.sentence_text))

    return [
        relative_position,
        reverse_position,
        float(token_count),
        normalized_length,
        named_entity_count,
        has_number,
    ]


class SentenceFeatureExtractor:
    """Fit and transform sentence examples into ML-ready feature matrices."""

    def __init__(self, config: SentenceFeatureConfig | None = None) -> None:
        """Initialize the feature extractor and its TF-IDF vectorizer."""

        self.config = config or SentenceFeatureConfig()
        self.vectorizer = None
        self.feature_names_: List[str] = []
        self._initialize_vectorizer()

    def _initialize_vectorizer(self) -> None:
        """Create the TF-IDF vectorizer lazily to keep imports lightweight."""

        if not self.config.use_tfidf:
            return

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as error:
            raise ImportError(
                "The 'scikit-learn' package is required for TF-IDF feature extraction. "
                "Install it with 'pip install scikit-learn'."
            ) from error

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=self.config.max_tfidf_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            ngram_range=self.config.ngram_range,
        )

    def fit(self, examples: Sequence[SentenceClassificationExample]) -> "SentenceFeatureExtractor":
        """Fit the TF-IDF vocabulary on sentence text."""

        texts = [example.sentence_text for example in examples]
        if self.vectorizer is not None:
            self.vectorizer.fit(texts)
        self.feature_names_ = self.get_feature_names()
        return self

    def transform(self, examples: Sequence[SentenceClassificationExample]):
        """Transform sentence examples into a combined sparse feature matrix."""

        if self.vectorizer is not None and not hasattr(self.vectorizer, "vocabulary_"):
            raise ValueError("The TF-IDF vectorizer must be fitted before calling transform().")
        if not examples:
            raise ValueError("Examples cannot be empty.")

        dense_matrix = np.asarray([compute_dense_features(example) for example in examples], dtype=float)

        if self.vectorizer is None:
            self.feature_names_ = self.get_feature_names()
            return dense_matrix

        try:
            from scipy.sparse import csr_matrix, hstack
        except ImportError as error:
            raise ImportError(
                "The 'scipy' package is required to combine sparse TF-IDF features. "
                "Install it with 'pip install scipy'."
            ) from error

        tfidf_matrix = self.vectorizer.transform([example.sentence_text for example in examples])
        dense_sparse = csr_matrix(dense_matrix)
        self.feature_names_ = self.get_feature_names()
        return hstack([tfidf_matrix, dense_sparse], format="csr")

    def fit_transform(self, examples: Sequence[SentenceClassificationExample]):
        """Fit the feature extractor and transform the provided examples."""

        return self.fit(examples).transform(examples)

    def extract_labels(self, examples: Sequence[SentenceClassificationExample]) -> np.ndarray:
        """Extract binary labels for supervised training."""

        return np.asarray([example.label for example in examples], dtype=int)

    def get_feature_names(self) -> List[str]:
        """Return combined TF-IDF and dense feature names."""

        dense_feature_names = [
            "relative_position",
            "reverse_position",
            "token_count",
            "normalized_length",
            "named_entity_count",
            "has_number",
        ]
        if self.vectorizer is None:
            return dense_feature_names
        if not hasattr(self.vectorizer, "vocabulary_"):
            return dense_feature_names
        tfidf_names = list(self.vectorizer.get_feature_names_out())
        return tfidf_names + dense_feature_names


__all__ = [
    "SentenceFeatureConfig",
    "SentenceFeatureExtractor",
    "compute_dense_features",
    "count_named_entities",
]
