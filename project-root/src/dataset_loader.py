"""Dataset loading and pseudo-label generation utilities for extractive summarization.

This module focuses on classical supervised extractive summarization workflows.
It provides:

1. A lightweight loader for the CNN/DailyMail dataset.
2. Sentence-level pseudo-label generation using ROUGE-style heuristics.
3. Reusable dataclasses that can feed a classical ML training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence

from .data_pipeline import clean_sentence_text, is_boilerplate_sentence, sentence_tokenize, word_tokenize


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
    prefer_local_cache: bool = True
    allow_remote_fallback: bool = False


@dataclass(frozen=True)
class SentenceLabelingConfig:
    """Configuration for sentence-level pseudo-label generation."""

    label_threshold: float = 0.18
    min_sentence_tokens: int = 4
    min_positive_sentences: int = 1
    max_positive_fraction: float = 0.35
    rouge1_weight: float = 0.35
    rouge2_weight: float = 0.20
    rouge_l_weight: float = 0.45

    def __post_init__(self) -> None:
        """Validate sentence labeling settings."""

        if not 0.0 <= self.label_threshold <= 1.0:
            raise ValueError("label_threshold must be between 0.0 and 1.0.")
        if self.min_sentence_tokens < 1:
            raise ValueError("min_sentence_tokens must be at least 1.")
        if self.min_positive_sentences < 0:
            raise ValueError("min_positive_sentences cannot be negative.")
        if not 0.0 < self.max_positive_fraction <= 1.0:
            raise ValueError("max_positive_fraction must be between 0.0 and 1.0.")
        total_weight = self.rouge1_weight + self.rouge2_weight + self.rouge_l_weight
        if total_weight <= 0.0:
            raise ValueError("At least one ROUGE weight must be positive.")


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
    rouge2_f1: float = 0.0
    rouge_l_f1: float = 0.0
    label_score: float = 0.0
    original_sentence_index: int = 0


@dataclass(frozen=True)
class ScoredSentence:
    """Intermediate container used for pseudo-label generation."""

    sentence_index: int
    sentence_text: str
    rouge1_f1: float
    rouge2_f1: float
    rouge_l_f1: float
    label_score: float


def _safe_record_id(record: Mapping[str, object], index: int, id_field: str) -> str:
    """Return a stable record identifier from a dataset row."""

    record_id = record.get(id_field)
    if record_id is None:
        return f"sample-{index}"
    return str(record_id)


def _require_datasets_library():
    """Import datasets objects lazily with a friendly error message."""

    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as error:
        raise ImportError(
            "The 'datasets' package is required to load CNN/DailyMail. "
            "Install it with 'pip install datasets'."
        ) from error
    return DownloadConfig, load_dataset


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


def compute_label_score(
    rouge1_f1: float,
    rouge2_f1: float,
    rouge_l_f1: float,
    config: SentenceLabelingConfig,
) -> float:
    """Combine ROUGE signals into a single label score."""

    total_weight = config.rouge1_weight + config.rouge2_weight + config.rouge_l_weight
    weighted_score = (
        config.rouge1_weight * rouge1_f1
        + config.rouge2_weight * rouge2_f1
        + config.rouge_l_weight * rouge_l_f1
    )
    return weighted_score / total_weight


class CNNDailyMailDatasetLoader:
    """Load article-summary pairs from the CNN/DailyMail dataset."""

    def __init__(self, config: CNNDailyMailLoaderConfig | None = None) -> None:
        """Initialize the dataset loader."""

        self.config = config or CNNDailyMailLoaderConfig()

    def load_records(self) -> List[ArticleSummaryRecord]:
        """Load a dataset split into a list of article-summary records."""

        if self.config.prefer_local_cache:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"

        DownloadConfig, load_dataset = _require_datasets_library()
        dataset = None

        if self.config.prefer_local_cache:
            try:
                from datasets import config as datasets_config

                datasets_config.HF_DATASETS_OFFLINE = True
            except Exception:
                pass

            try:
                dataset = load_dataset(
                    self.config.dataset_name,
                    self.config.dataset_version,
                    split=self.config.split,
                    streaming=self.config.streaming,
                    download_config=DownloadConfig(local_files_only=True),
                )
            except Exception:
                dataset = None
                if not self.config.allow_remote_fallback:
                    raise

        if dataset is None:
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


def _collect_scored_sentences(
    record: ArticleSummaryRecord,
    config: SentenceLabelingConfig,
) -> List[ScoredSentence]:
    """Create scored sentences for one article before label assignment."""

    scored_sentences: List[ScoredSentence] = []
    for sentence_index, sentence_text in enumerate(sentence_tokenize(record.article_text)):
        cleaned_sentence = clean_sentence_text(sentence_text)
        if not cleaned_sentence:
            continue
        if is_boilerplate_sentence(cleaned_sentence, config.min_sentence_tokens):
            continue
        if len(word_tokenize(cleaned_sentence)) < config.min_sentence_tokens:
            continue

        rouge1_f1 = rouge_n_f1(cleaned_sentence, record.summary_text, n=1)
        rouge2_f1 = rouge_n_f1(cleaned_sentence, record.summary_text, n=2)
        rouge_l_value = rouge_l_f1(cleaned_sentence, record.summary_text)
        label_score = compute_label_score(rouge1_f1, rouge2_f1, rouge_l_value, config)
        scored_sentences.append(
            ScoredSentence(
                sentence_index=sentence_index,
                sentence_text=cleaned_sentence,
                rouge1_f1=rouge1_f1,
                rouge2_f1=rouge2_f1,
                rouge_l_f1=rouge_l_value,
                label_score=label_score,
            )
        )
    return scored_sentences


def _select_positive_sentence_indices(
    scored_sentences: Sequence[ScoredSentence],
    config: SentenceLabelingConfig,
) -> set[int]:
    """Choose which sentence indices should receive positive labels."""

    if not scored_sentences:
        return set()

    sorted_indices = sorted(
        range(len(scored_sentences)),
        key=lambda index: scored_sentences[index].label_score,
        reverse=True,
    )
    required_positive = min(config.min_positive_sentences, len(scored_sentences))
    maximum_positive = max(required_positive, math.ceil(len(scored_sentences) * config.max_positive_fraction))

    positive_indices = {
        index for index, sentence in enumerate(scored_sentences) if sentence.label_score >= config.label_threshold
    }
    positive_indices.update(sorted_indices[:required_positive])
    capped_positive = set(sorted_indices[:maximum_positive])
    return positive_indices.intersection(capped_positive)


def generate_sentence_labels(
    record: ArticleSummaryRecord,
    config: SentenceLabelingConfig | None = None,
) -> List[SentenceClassificationExample]:
    """Convert one article-summary pair into sentence-level classification examples."""

    labeling_config = config or SentenceLabelingConfig()
    scored_sentences = _collect_scored_sentences(record, labeling_config)
    positive_indices = _select_positive_sentence_indices(scored_sentences, labeling_config)
    sentence_count = len(scored_sentences)

    return [
        SentenceClassificationExample(
            article_id=record.article_id,
            sentence_index=index,
            sentence_count=sentence_count,
            sentence_text=sentence.sentence_text,
            article_text=record.article_text,
            reference_summary=record.summary_text,
            label=int(index in positive_indices),
            rouge1_f1=sentence.rouge1_f1,
            rouge2_f1=sentence.rouge2_f1,
            rouge_l_f1=sentence.rouge_l_f1,
            label_score=sentence.label_score,
            original_sentence_index=sentence.sentence_index,
        )
        for index, sentence in enumerate(scored_sentences)
    ]


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
    "compute_label_score",
    "generate_sentence_labels",
    "rouge_l_f1",
    "rouge_n_f1",
]
