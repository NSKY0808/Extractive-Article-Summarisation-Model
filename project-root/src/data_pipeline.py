"""Data preprocessing pipeline for clustered news summarization inputs.

This module implements the extractive compression stage that prepares
multi-document news clusters for downstream abstractive summarization.
The pipeline:

1. Splits article text into sentences.
2. Removes boilerplate and near-duplicate sentences.
3. Ranks the remaining sentences with TF-IDF centroid similarity.
4. Selects the top portion of sentences and returns compressed text.

The implementation intentionally avoids external NLP runtime dependencies so the
pipeline can run in lightweight environments and remain easy to test.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Dict, List, Mapping, Sequence, Tuple


WORD_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9'-]*\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=(?:[\"'(\[])?[A-Z0-9])")
NAMED_ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
MULTISPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class SentenceRecord:
    """Container for sentence text and its position inside a cluster."""

    text: str
    article_index: int
    sentence_index: int


@dataclass(frozen=True)
class DataPipelineConfig:
    """Configuration values for the cluster preprocessing pipeline."""

    duplicate_similarity_threshold: float = 0.8
    top_sentence_fraction: float = 0.3
    min_word_count: int = 4
    named_entity_bonus: float = 0.05
    max_named_entity_bonus: float = 0.15

    def __post_init__(self) -> None:
        """Validate configuration ranges to prevent invalid pipeline settings."""

        if not 0.0 <= self.duplicate_similarity_threshold <= 1.0:
            raise ValueError("duplicate_similarity_threshold must be between 0.0 and 1.0.")
        if not 0.0 < self.top_sentence_fraction <= 1.0:
            raise ValueError("top_sentence_fraction must be between 0.0 and 1.0.")
        if self.min_word_count < 1:
            raise ValueError("min_word_count must be at least 1.")
        if self.named_entity_bonus < 0.0 or self.max_named_entity_bonus < 0.0:
            raise ValueError("Named entity bonuses must be non-negative.")
        if self.named_entity_bonus > self.max_named_entity_bonus:
            raise ValueError("named_entity_bonus cannot exceed max_named_entity_bonus.")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim surrounding spacing."""

    return MULTISPACE_PATTERN.sub(" ", text).strip()


def clean_sentence_text(sentence: str) -> str:
    """Normalize sentence spacing and strip wrapper punctuation."""

    cleaned = normalize_whitespace(sentence)
    return cleaned.strip(" -\t\r\n")


@lru_cache(maxsize=50000)
def word_tokenize(text: str) -> List[str]:
    """Tokenize a text span into lowercase word tokens."""

    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


@lru_cache(maxsize=20000)
def sentence_tokenize(text: str) -> List[str]:
    """Split article text into sentence-like spans using punctuation cues."""

    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    candidate_sentences = SENTENCE_SPLIT_PATTERN.split(normalized)
    sentences = [clean_sentence_text(sentence) for sentence in candidate_sentences]
    return [sentence for sentence in sentences if sentence]


def contains_named_entity_hint(sentence: str) -> bool:
    """Detect whether a sentence appears to contain named entities."""

    return bool(NAMED_ENTITY_PATTERN.search(sentence))


def is_boilerplate_sentence(sentence: str, min_word_count: int) -> bool:
    """Heuristically identify low-value or templated news boilerplate."""

    normalized = sentence.lower()
    boilerplate_markers = (
        "all rights reserved",
        "this material may not be published",
        "click here",
        "follow us",
        "subscribe",
        "newsletter",
        "advertisement",
        "sign up",
        "copyright",
    )
    if any(marker in normalized for marker in boilerplate_markers):
        return True

    if len(word_tokenize(sentence)) < min_word_count:
        return True

    alpha_ratio = sum(character.isalpha() for character in sentence) / max(len(sentence), 1)
    return alpha_ratio < 0.5


def collect_cluster_sentences(
    articles: Sequence[str],
    config: DataPipelineConfig,
) -> List[SentenceRecord]:
    """Tokenize cluster articles and drop boilerplate sentences."""

    collected: List[SentenceRecord] = []
    for article_index, article in enumerate(articles):
        for sentence_index, sentence in enumerate(sentence_tokenize(article)):
            cleaned_sentence = clean_sentence_text(sentence)
            if not cleaned_sentence:
                continue
            if is_boilerplate_sentence(cleaned_sentence, config.min_word_count):
                continue
            collected.append(
                SentenceRecord(
                    text=cleaned_sentence,
                    article_index=article_index,
                    sentence_index=sentence_index,
                )
            )
    return collected


def compute_document_frequencies(tokenized_texts: Sequence[Sequence[str]]) -> Counter[str]:
    """Count how many sentence-documents contain each token."""

    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized_texts:
        document_frequencies.update(set(tokens))
    return document_frequencies


def build_tfidf_vectors(texts: Sequence[str]) -> List[Dict[str, float]]:
    """Create sparse TF-IDF vectors for sentence-level texts."""

    tokenized_texts = [word_tokenize(text) for text in texts]
    document_count = len(tokenized_texts)
    if document_count == 0:
        return []

    document_frequencies = compute_document_frequencies(tokenized_texts)
    vectors: List[Dict[str, float]] = []

    for tokens in tokenized_texts:
        if not tokens:
            vectors.append({})
            continue

        token_counts = Counter(tokens)
        sentence_length = len(tokens)
        vector: Dict[str, float] = {}
        for token, count in token_counts.items():
            term_frequency = count / sentence_length
            inverse_document_frequency = math.log((1 + document_count) / (1 + document_frequencies[token])) + 1.0
            vector[token] = term_frequency * inverse_document_frequency
        vectors.append(vector)

    return vectors


def cosine_similarity(vector_a: Mapping[str, float], vector_b: Mapping[str, float]) -> float:
    """Compute cosine similarity for sparse TF-IDF vectors."""

    if not vector_a or not vector_b:
        return 0.0

    shared_terms = set(vector_a).intersection(vector_b)
    numerator = sum(vector_a[term] * vector_b[term] for term in shared_terms)
    norm_a = math.sqrt(sum(weight * weight for weight in vector_a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in vector_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def remove_duplicate_sentences(
    sentences: Sequence[SentenceRecord],
    similarity_threshold: float,
) -> List[SentenceRecord]:
    """Greedily remove near-duplicate sentences while preserving early context."""

    if not sentences:
        return []

    vectors = build_tfidf_vectors([sentence.text for sentence in sentences])
    kept_sentences: List[SentenceRecord] = []
    kept_indices: List[int] = []

    for index, sentence in enumerate(sentences):
        is_duplicate = any(
            cosine_similarity(vectors[index], vectors[kept_index]) >= similarity_threshold
            for kept_index in kept_indices
        )
        if is_duplicate:
            continue
        kept_sentences.append(sentence)
        kept_indices.append(index)

    return kept_sentences


def build_centroid_vector(vectors: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Average sentence vectors into a single centroid representation."""

    if not vectors:
        return {}

    centroid: Counter[str] = Counter()
    for vector in vectors:
        centroid.update(vector)

    vector_count = len(vectors)
    return {token: weight / vector_count for token, weight in centroid.items()}


def score_sentence(
    sentence: SentenceRecord,
    vector: Mapping[str, float],
    centroid_vector: Mapping[str, float],
    config: DataPipelineConfig,
) -> float:
    """Score a sentence using centroid similarity and a small entity bonus."""

    base_score = cosine_similarity(vector, centroid_vector)
    if not contains_named_entity_hint(sentence.text):
        return base_score

    entity_count = len(NAMED_ENTITY_PATTERN.findall(sentence.text))
    entity_bonus = min(entity_count * config.named_entity_bonus, config.max_named_entity_bonus)
    return base_score + entity_bonus


def rank_sentences(
    sentences: Sequence[SentenceRecord],
    config: DataPipelineConfig,
) -> List[Tuple[SentenceRecord, float]]:
    """Rank sentences by TF-IDF centroid relevance."""

    if not sentences:
        return []

    vectors = build_tfidf_vectors([sentence.text for sentence in sentences])
    centroid_vector = build_centroid_vector(vectors)
    ranked = [
        (sentence, score_sentence(sentence, vector, centroid_vector, config))
        for sentence, vector in zip(sentences, vectors)
    ]
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def select_top_sentences(
    ranked_sentences: Sequence[Tuple[SentenceRecord, float]],
    top_sentence_fraction: float,
) -> List[SentenceRecord]:
    """Select the top fraction of ranked sentences and restore narrative order."""

    if not ranked_sentences:
        return []

    selection_count = max(1, math.ceil(len(ranked_sentences) * top_sentence_fraction))
    selected = [sentence for sentence, _ in ranked_sentences[:selection_count]]
    return sorted(selected, key=lambda sentence: (sentence.article_index, sentence.sentence_index))


def compress_cluster(
    articles: Sequence[str],
    config: DataPipelineConfig | None = None,
) -> Dict[str, str]:
    """Convert a cluster of articles into compressed model-ready text."""

    pipeline_config = config or DataPipelineConfig()
    collected_sentences = collect_cluster_sentences(articles, pipeline_config)
    unique_sentences = remove_duplicate_sentences(
        collected_sentences,
        similarity_threshold=pipeline_config.duplicate_similarity_threshold,
    )
    ranked_sentences = rank_sentences(unique_sentences, pipeline_config)
    selected_sentences = select_top_sentences(
        ranked_sentences,
        top_sentence_fraction=pipeline_config.top_sentence_fraction,
    )
    input_text = " ".join(sentence.text for sentence in selected_sentences)
    return {"input_text": input_text}


class ClusterPreprocessor:
    """Reusable pipeline object for cluster-level text compression."""

    def __init__(self, config: DataPipelineConfig | None = None) -> None:
        """Initialize the preprocessor with optional pipeline configuration."""

        self.config = config or DataPipelineConfig()

    def process_cluster(self, articles: Sequence[str]) -> Dict[str, str]:
        """Run the configured preprocessing pipeline on a news cluster."""

        return compress_cluster(articles, config=self.config)


__all__ = [
    "ClusterPreprocessor",
    "DataPipelineConfig",
    "SentenceRecord",
    "clean_sentence_text",
    "collect_cluster_sentences",
    "compress_cluster",
    "cosine_similarity",
    "rank_sentences",
    "remove_duplicate_sentences",
    "select_top_sentences",
    "sentence_tokenize",
    "word_tokenize",
]
