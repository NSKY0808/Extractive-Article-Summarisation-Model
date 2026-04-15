"""Core modules for the extractive summarization project."""

from .data_pipeline import sentence_tokenize, word_tokenize
from .dataset_loader import (
    ArticleSummaryRecord,
    CNNDailyMailDatasetLoader,
    CNNDailyMailLoaderConfig,
    SentenceClassificationExample,
    SentenceLabelingConfig,
    build_sentence_classification_dataset,
    generate_sentence_labels,
)
from .feature_pipeline import SentenceFeatureConfig, SentenceFeatureExtractor

__all__ = [
    "ArticleSummaryRecord",
    "CNNDailyMailDatasetLoader",
    "CNNDailyMailLoaderConfig",
    "SentenceClassificationExample",
    "SentenceFeatureConfig",
    "SentenceFeatureExtractor",
    "SentenceLabelingConfig",
    "build_sentence_classification_dataset",
    "generate_sentence_labels",
    "sentence_tokenize",
    "word_tokenize",
]
