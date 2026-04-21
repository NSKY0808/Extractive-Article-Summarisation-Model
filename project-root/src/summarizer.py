"""Inference-time summarization pipeline for classical extractive models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from models.extractive_classifier import ExtractiveSentenceClassifier

from .data_pipeline import (
    build_tfidf_vectors,
    clean_sentence_text,
    cosine_similarity,
    is_boilerplate_sentence,
    sentence_tokenize,
    word_tokenize,
)
from .dataset_loader import SentenceClassificationExample


@dataclass(frozen=True)
class SummarizationConfig:
    """Configuration for extractive summary generation."""

    top_n_sentences: int = 3
    redundancy_threshold: float = 0.8
    min_sentence_tokens: int = 4
    mmr_lambda: float = 0.85
    max_candidates: int = 15

    def __post_init__(self) -> None:
        """Validate summarization settings."""

        if self.top_n_sentences < 1:
            raise ValueError("top_n_sentences must be at least 1.")
        if not 0.0 <= self.redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be between 0.0 and 1.0.")
        if self.min_sentence_tokens < 1:
            raise ValueError("min_sentence_tokens must be at least 1.")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0.0 and 1.0.")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1.")


@dataclass(frozen=True)
class RankedSentence:
    """Sentence text paired with ranking metadata."""

    sentence_index: int
    sentence_text: str
    score: float


def build_inference_examples(article_text: str, min_sentence_tokens: int = 4) -> List[SentenceClassificationExample]:
    """Convert article text into sentence-level examples for inference."""

    raw_sentences = sentence_tokenize(article_text)
    cleaned_sentences: List[tuple[int, str]] = []

    for sentence_index, sentence_text in enumerate(raw_sentences):
        cleaned_sentence = clean_sentence_text(sentence_text)
        if not cleaned_sentence:
            continue
        if is_boilerplate_sentence(cleaned_sentence, min_sentence_tokens):
            continue
        if len(word_tokenize(cleaned_sentence)) < min_sentence_tokens:
            continue
        cleaned_sentences.append((sentence_index, cleaned_sentence))

    sentence_count = len(cleaned_sentences)
    return [
        SentenceClassificationExample(
            article_id="inference",
            sentence_index=position_index,
            sentence_count=sentence_count,
            sentence_text=sentence_text,
            article_text=article_text,
            reference_summary="",
            label=0,
            rouge1_f1=0.0,
            rouge2_f1=0.0,
            rouge_l_f1=0.0,
            label_score=0.0,
            original_sentence_index=sentence_index,
        )
        for position_index, (sentence_index, sentence_text) in enumerate(cleaned_sentences)
    ]


def rank_article_sentences(
    article_text: str,
    classifier: ExtractiveSentenceClassifier,
    config: SummarizationConfig | None = None,
) -> List[RankedSentence]:
    """Score and rank article sentences by predicted importance."""

    summarization_config = config or SummarizationConfig()
    examples = build_inference_examples(article_text, summarization_config.min_sentence_tokens)
    if not examples:
        return []

    scores = classifier.predict_scores(examples)
    ranked = [
        RankedSentence(
            sentence_index=example.original_sentence_index,
            sentence_text=example.sentence_text,
            score=float(score),
        )
        for example, score in zip(examples, scores)
    ]
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def remove_redundant_ranked_sentences(
    ranked_sentences: Sequence[RankedSentence],
    redundancy_threshold: float,
) -> List[RankedSentence]:
    """Remove highly similar ranked sentences while keeping higher-scored ones."""

    if not ranked_sentences:
        return []

    vectors = build_tfidf_vectors([sentence.sentence_text for sentence in ranked_sentences])
    kept_sentences: List[RankedSentence] = []
    kept_indices: List[int] = []

    for index, ranked_sentence in enumerate(ranked_sentences):
        is_redundant = any(
            cosine_similarity(vectors[index], vectors[kept_index]) >= redundancy_threshold
            for kept_index in kept_indices
        )
        if is_redundant:
            continue
        kept_sentences.append(ranked_sentence)
        kept_indices.append(index)

    return kept_sentences


def select_summary_sentences(
    ranked_sentences: Sequence[RankedSentence],
    config: SummarizationConfig,
) -> List[RankedSentence]:
    """Select summary sentences using maximal marginal relevance."""

    if not ranked_sentences:
        return []

    candidate_sentences = list(ranked_sentences[: config.max_candidates])
    vectors = build_tfidf_vectors([sentence.sentence_text for sentence in candidate_sentences])
    selected_indices: List[int] = []
    remaining_indices = list(range(len(candidate_sentences)))

    while remaining_indices and len(selected_indices) < config.top_n_sentences:
        best_index = None
        best_score = float("-inf")

        for candidate_index in remaining_indices:
            candidate_score = candidate_sentences[candidate_index].score
            if not selected_indices:
                mmr_score = candidate_score
            else:
                max_similarity = max(
                    cosine_similarity(vectors[candidate_index], vectors[selected_index])
                    for selected_index in selected_indices
                )
                if max_similarity >= config.redundancy_threshold:
                    continue
                mmr_score = config.mmr_lambda * candidate_score - (1.0 - config.mmr_lambda) * max_similarity

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = candidate_index

        if best_index is None:
            break

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    selected_sentences = [candidate_sentences[index] for index in selected_indices]
    return sorted(selected_sentences, key=lambda item: item.sentence_index)


def summarize_article(
    article_text: str,
    classifier: ExtractiveSentenceClassifier,
    config: SummarizationConfig | None = None,
) -> Dict[str, object]:
    """Generate an extractive summary for one article."""

    summarization_config = config or SummarizationConfig()
    ranked_sentences = rank_article_sentences(article_text, classifier, summarization_config)
    filtered_sentences = remove_redundant_ranked_sentences(
        ranked_sentences,
        redundancy_threshold=min(0.98, summarization_config.redundancy_threshold + 0.15),
    )
    selected_sentences = select_summary_sentences(filtered_sentences, summarization_config)
    summary_text = " ".join(sentence.sentence_text for sentence in selected_sentences)

    return {
        "summary_text": summary_text,
        "ranked_sentences": ranked_sentences,
        "selected_sentences": selected_sentences,
    }


__all__ = [
    "RankedSentence",
    "SummarizationConfig",
    "build_inference_examples",
    "rank_article_sentences",
    "remove_redundant_ranked_sentences",
    "select_summary_sentences",
    "summarize_article",
]