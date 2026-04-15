"""Dataset loading and pseudo-label generation utilities for extractive summarization.

This module focuses on classical supervised extractive summarization workflows.
It provides:

1. A lightweight loader for the CNN/DailyMail dataset.
2. Sentence-level pseudo-label generation using ROUGE-style heuristics.
3. Reusable dataclasses that can feed a classical ML training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence

from .data_pipeline import sentence_tokenize, word_tokenize


@dataclass(frozen=True)
class CNNDailyMailLoaderConfig:
    """Configuration for loading CNN/DailyMail article-summary pairs."""

    dataset_name: str = "cnn_dailymail"
    dataset_version: str = "3.0.0"
    split: str = "train"
    sample_limit: Optional[int] = None
    streaming: bool = False
    article_field: str = "article"
    summary_field: str = "highlights"
    id_field: str = "id"


@dataclass(frozen=True)
class SentenceLabelingConfig:
    """Configuration for sentence-level pseudo-label generation."""

    rouge_threshold: float = 0.2
    min_sentence_tokens: int = 4

    def __post_init__(self) -> None:
        """Validate sentence labeling settings."""

        if not 0.0 <= self.rouge_threshold <= 1.0:
            raise ValueError("rouge_threshold must be between 0.0 and 1.0.")
        if self.min_sentence_tokens < 1:
            raise ValueError("min_sentence_tokens must be at least 1.")


@dataclass(frozen=True)
class ArticleSummaryRecord:
    """Container for one article-summary training example."""

    article_id: str
    article_text: str
    summary_text: str
    split: str


@dataclass(frozen=True)
class SentenceClassificationExample:
    """Sentence-level supervised example for importance classification."""

    article_id: str
    sentence_index: int
    sentence_count: int
    sentence_text: str
    article_text: str
    reference_summary: str
    label: int
    rouge1_f1: float
    rouge_l_f1: float


def _safe_record_id(record: Mapping[str, object], index: int, id_field: str) -> str:
    """Return a stable record identifier from a dataset row."""

    record_id = record.get(id_field)
    if record_id is None:
        return f"sample-{index}"
    return str(record_id)


def _require_datasets_library():
    """Import the datasets library lazily with a friendly error message."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "The 'datasets' package is required to load CNN/DailyMail. "
            "Install it with 'pip install datasets'."
        ) from error
    return load_dataset


def _generate_ngrams(tokens: Sequence[str], n: int) -> List[tuple[str, ...]]:
    """Generate n-grams from a token sequence."""

    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _lcs_length(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
    """Compute the longest common subsequence length for two token lists."""

    if not tokens_a or not tokens_b:
        return 0

    previous_row = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        current_row = [0]
        for column_index, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                current_row.append(previous_row[column_index - 1] + 1)
            else:
                current_row.append(max(previous_row[column_index], current_row[-1]))
        previous_row = current_row
    return previous_row[-1]


def rouge_n_f1(candidate_text: str, reference_text: str, n: int = 1) -> float:
    """Compute ROUGE-N F1 using token overlap."""

    candidate_tokens = word_tokenize(candidate_text)
    reference_tokens = word_tokenize(reference_text)
    candidate_ngrams = _generate_ngrams(candidate_tokens, n)
    reference_ngrams = _generate_ngrams(reference_tokens, n)

    if not candidate_ngrams or not reference_ngrams:
        return 0.0

    reference_counts = {}
    for ngram in reference_ngrams:
        reference_counts[ngram] = reference_counts.get(ngram, 0) + 1

    overlap = 0
    for ngram in candidate_ngrams:
        remaining = reference_counts.get(ngram, 0)
        if remaining > 0:
            overlap += 1
            reference_counts[ngram] = remaining - 1

    precision = overlap / len(candidate_ngrams)
    recall = overlap / len(reference_ngrams)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(candidate_text: str, reference_text: str) -> float:
    """Compute ROUGE-L F1 using longest common subsequence overlap."""

    candidate_tokens = word_tokenize(candidate_text)
    reference_tokens = word_tokenize(reference_text)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    lcs = _lcs_length(candidate_tokens, reference_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class CNNDailyMailDatasetLoader:
    """Load article-summary pairs from the CNN/DailyMail dataset."""

    def __init__(self, config: CNNDailyMailLoaderConfig | None = None) -> None:
        """Initialize the dataset loader."""

        self.config = config or CNNDailyMailLoaderConfig()

    def load_records(self) -> List[ArticleSummaryRecord]:
        """Load a dataset split into a list of article-summary records."""

        load_dataset = _require_datasets_library()
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_version,
            split=self.config.split,
            streaming=self.config.streaming,
        )

        records: List[ArticleSummaryRecord] = []
        for index, record in enumerate(dataset):
            article_text = str(record[self.config.article_field]).strip()
            summary_text = str(record[self.config.summary_field]).strip()
            if not article_text or not summary_text:
                continue

            records.append(
                ArticleSummaryRecord(
                    article_id=_safe_record_id(record, index, self.config.id_field),
                    article_text=article_text,
                    summary_text=summary_text,
                    split=self.config.split,
                )
            )

            if self.config.sample_limit is not None and len(records) >= self.config.sample_limit:
                break

        return records

    def iter_records(self) -> Iterator[ArticleSummaryRecord]:
        """Yield records one by one to support streaming workflows."""

        for record in self.load_records():
            yield record


def generate_sentence_labels(
    record: ArticleSummaryRecord,
    config: SentenceLabelingConfig | None = None,
) -> List[SentenceClassificationExample]:
    """Convert one article-summary pair into sentence-level classification examples."""

    labeling_config = config or SentenceLabelingConfig()
    sentences = sentence_tokenize(record.article_text)
    sentence_count = len(sentences)
    examples: List[SentenceClassificationExample] = []

    for sentence_index, sentence_text in enumerate(sentences):
        if len(word_tokenize(sentence_text)) < labeling_config.min_sentence_tokens:
            continue

        rouge1 = rouge_n_f1(sentence_text, record.summary_text, n=1)
        rouge_l = rouge_l_f1(sentence_text, record.summary_text)
        label = int(max(rouge1, rouge_l) >= labeling_config.rouge_threshold)
        examples.append(
            SentenceClassificationExample(
                article_id=record.article_id,
                sentence_index=sentence_index,
                sentence_count=sentence_count,
                sentence_text=sentence_text,
                article_text=record.article_text,
                reference_summary=record.summary_text,
                label=label,
                rouge1_f1=rouge1,
                rouge_l_f1=rouge_l,
            )
        )

    return examples


def build_sentence_classification_dataset(
    records: Iterable[ArticleSummaryRecord],
    config: SentenceLabelingConfig | None = None,
) -> List[SentenceClassificationExample]:
    """Convert article-summary records into a flat sentence classification dataset."""

    labeling_config = config or SentenceLabelingConfig()
    examples: List[SentenceClassificationExample] = []
    for record in records:
        examples.extend(generate_sentence_labels(record, config=labeling_config))
    return examples


__all__ = [
    "ArticleSummaryRecord",
    "CNNDailyMailDatasetLoader",
    "CNNDailyMailLoaderConfig",
    "SentenceClassificationExample",
    "SentenceLabelingConfig",
    "build_sentence_classification_dataset",
    "generate_sentence_labels",
    "rouge_l_f1",
    "rouge_n_f1",
]
