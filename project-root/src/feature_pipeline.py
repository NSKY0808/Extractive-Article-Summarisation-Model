"""Feature engineering utilities for classical extractive summarization models.

This module combines sparse TF-IDF sentence representations with small,
interpretable numeric features such as sentence position and length. The output
is designed for classical binary classifiers like Logistic Regression, SVM, or
Random Forest.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .data_pipeline import NAMED_ENTITY_PATTERN, word_tokenize
from .dataset_loader import SentenceClassificationExample


@dataclass(frozen=True)
class SentenceFeatureConfig:
    """Configuration for sentence-level feature extraction."""

    max_tfidf_features: int = 8000
    min_df: int = 2
    max_df: float = 0.9
    ngram_range: tuple[int, int] = (1, 2)
    use_tfidf: bool = True
    scale_dense_features: bool = True

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


DENSE_FEATURE_NAMES = [
    "relative_position",
    "reverse_position",
    "lead_bias",
    "is_first_sentence",
    "is_second_sentence",
    "is_last_sentence",
    "token_count",
    "normalized_length",
    "sentence_to_article_ratio",
    "named_entity_count",
    "contains_named_entity",
    "has_number",
    "contains_quote",
    "contains_colon",
    "capitalized_token_ratio",
    "article_frequency_score",
]


def count_named_entities(sentence: str) -> int:
    """Count simple named-entity hints using capitalized phrase matching."""

    return len(NAMED_ENTITY_PATTERN.findall(sentence))


def _build_article_frequency_score(tokens: Sequence[str], article_tokens: Sequence[str]) -> float:
    """Estimate how central a sentence is using article token frequencies."""

    if not tokens or not article_tokens:
        return 0.0

    article_counts = Counter(article_tokens)
    unique_tokens = set(tokens)
    return sum(article_counts[token] for token in unique_tokens) / (len(unique_tokens) * len(article_tokens))


def compute_dense_features(example: SentenceClassificationExample) -> List[float]:
    """Build small explainable numeric features for one sentence."""

    tokens = word_tokenize(example.sentence_text)
    article_tokens = word_tokenize(example.article_text)
    token_count = len(tokens)
    sentence_count = max(example.sentence_count, 1)
    denominator = max(sentence_count - 1, 1)

    relative_position = example.sentence_index / denominator
    reverse_position = (sentence_count - example.sentence_index - 1) / denominator
    lead_bias = 1.0 / (example.sentence_index + 1)
    normalized_length = token_count / 40.0
    sentence_to_article_ratio = token_count / max(len(article_tokens), 1)
    named_entity_count = float(count_named_entities(example.sentence_text))
    contains_named_entity = float(named_entity_count > 0)
    has_number = float(any(character.isdigit() for character in example.sentence_text))
    contains_quote = float('"' in example.sentence_text or "'" in example.sentence_text)
    contains_colon = float(":" in example.sentence_text)
    capitalized_token_ratio = sum(token[:1].isupper() for token in example.sentence_text.split()) / max(
        len(example.sentence_text.split()),
        1,
    )
    article_frequency_score = _build_article_frequency_score(tokens, article_tokens)

    return [
        relative_position,
        reverse_position,
        lead_bias,
        float(example.sentence_index == 0),
        float(example.sentence_index == 1),
        float(example.sentence_index == sentence_count - 1),
        float(token_count),
        normalized_length,
        sentence_to_article_ratio,
        named_entity_count,
        contains_named_entity,
        has_number,
        contains_quote,
        contains_colon,
        capitalized_token_ratio,
        article_frequency_score,
    ]


class SentenceFeatureExtractor:
    """Fit and transform sentence examples into ML-ready feature matrices."""

    def __init__(self, config: SentenceFeatureConfig | None = None) -> None:
        """Initialize the feature extractor and its TF-IDF vectorizer."""

        self.config = config or SentenceFeatureConfig()
        self.vectorizer = None
        self.dense_scaler = None
        self.feature_names_: List[str] = []
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Create the TF-IDF vectorizer and dense scaler lazily."""

        if self.config.use_tfidf:
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
                sublinear_tf=True,
                stop_words="english",
            )

        if self.config.scale_dense_features:
            try:
                from sklearn.preprocessing import StandardScaler
            except ImportError as error:
                raise ImportError(
                    "The 'scikit-learn' package is required for dense feature scaling. "
                    "Install it with 'pip install scikit-learn'."
                ) from error
            self.dense_scaler = StandardScaler()

    def fit(self, examples: Sequence[SentenceClassificationExample]) -> "SentenceFeatureExtractor":
        """Fit the TF-IDF vocabulary and dense feature scaler."""

        if not examples:
            raise ValueError("Examples cannot be empty.")

        texts = [example.sentence_text for example in examples]
        dense_matrix = np.asarray([compute_dense_features(example) for example in examples], dtype=float)

        vectorizer = getattr(self, "vectorizer", None)
        dense_scaler = getattr(self, "dense_scaler", None)

        if vectorizer is not None:
            vectorizer.fit(texts)
        if dense_scaler is not None:
            dense_scaler.fit(dense_matrix)
        self.feature_names_ = self.get_feature_names()
        return self

    def transform(self, examples: Sequence[SentenceClassificationExample]):
        """Transform sentence examples into a combined sparse feature matrix."""

        vectorizer = getattr(self, "vectorizer", None)
        dense_scaler = getattr(self, "dense_scaler", None)

        if vectorizer is not None and not hasattr(vectorizer, "vocabulary_"):
            raise ValueError("The TF-IDF vectorizer must be fitted before calling transform().")
        if dense_scaler is not None and not hasattr(dense_scaler, "mean_"):
            raise ValueError("The dense feature scaler must be fitted before calling transform().")
        if not examples:
            raise ValueError("Examples cannot be empty.")

        dense_matrix = np.asarray([compute_dense_features(example) for example in examples], dtype=float)
        if dense_scaler is not None:
            dense_matrix = dense_scaler.transform(dense_matrix)

        if vectorizer is None:
            self.feature_names_ = self.get_feature_names()
            return dense_matrix

        try:
            from scipy.sparse import csr_matrix, hstack
        except ImportError as error:
            raise ImportError(
                "The 'scipy' package is required to combine sparse TF-IDF features. "
                "Install it with 'pip install scipy'."
            ) from error

        tfidf_matrix = vectorizer.transform([example.sentence_text for example in examples])
        dense_sparse = csr_matrix(dense_matrix)
        self.feature_names_ = self.get_feature_names()
        return hstack([tfidf_matrix, dense_sparse], format="csr")

    def fit_transform(self, examples: Sequence[SentenceClassificationExample]):
        """Fit the feature extractor and transform the provided examples."""

        return self.fit(examples).transform(examples)

    def extract_labels(self, examples: Sequence[SentenceClassificationExample]) -> np.ndarray:
        """Extract binary labels for supervised training."""

        return np.asarray([example.label for example in examples], dtype=int)

    def extract_sample_weights(self, examples: Sequence[SentenceClassificationExample]) -> np.ndarray:
        """Build optional sample weights based on pseudo-label confidence."""

        weights = []
        for example in examples:
            base_weight = 1.0 + example.label_score
            if example.label == 1:
                base_weight += 0.5
            weights.append(base_weight)
        return np.asarray(weights, dtype=float)

    def get_feature_names(self) -> List[str]:
        """Return combined TF-IDF and dense feature names."""

        vectorizer = getattr(self, "vectorizer", None)
        if vectorizer is None or not hasattr(vectorizer, "vocabulary_"):
            return list(DENSE_FEATURE_NAMES)
        tfidf_names = list(vectorizer.get_feature_names_out())
        return tfidf_names + list(DENSE_FEATURE_NAMES)


__all__ = [
    "SentenceFeatureConfig",
    "SentenceFeatureExtractor",
    "compute_dense_features",
    "count_named_entities",
]