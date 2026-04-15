"""Evaluation helpers for extractive summarization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from models.extractive_classifier import ExtractiveSentenceClassifier

from .dataset_loader import ArticleSummaryRecord, rouge_l_f1, rouge_n_f1
from .summarizer import SummarizationConfig, summarize_article


@dataclass(frozen=True)
class RougeScore:
    """Container for common ROUGE metrics."""

    rouge_1: float
    rouge_2: float
    rouge_l: float


def evaluate_summary(generated_summary: str, reference_summary: str) -> RougeScore:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L for one summary pair."""

    return RougeScore(
        rouge_1=rouge_n_f1(generated_summary, reference_summary, n=1),
        rouge_2=rouge_n_f1(generated_summary, reference_summary, n=2),
        rouge_l=rouge_l_f1(generated_summary, reference_summary),
    )


def average_rouge_scores(scores: Sequence[RougeScore]) -> RougeScore:
    """Average ROUGE metrics across multiple examples."""

    if not scores:
        return RougeScore(rouge_1=0.0, rouge_2=0.0, rouge_l=0.0)

    count = len(scores)
    return RougeScore(
        rouge_1=sum(score.rouge_1 for score in scores) / count,
        rouge_2=sum(score.rouge_2 for score in scores) / count,
        rouge_l=sum(score.rouge_l for score in scores) / count,
    )


def evaluate_records(
    records: Iterable[ArticleSummaryRecord],
    classifier: ExtractiveSentenceClassifier,
    config: SummarizationConfig | None = None,
) -> Dict[str, object]:
    """Generate summaries for dataset records and report average ROUGE."""

    summarization_config = config or SummarizationConfig()
    scores: List[RougeScore] = []
    predictions: List[Mapping[str, object]] = []

    for record in records:
        prediction = summarize_article(record.article_text, classifier, summarization_config)
        score = evaluate_summary(prediction["summary_text"], record.summary_text)
        scores.append(score)
        predictions.append(
            {
                "article_id": record.article_id,
                "generated_summary": prediction["summary_text"],
                "reference_summary": record.summary_text,
                "rouge_1": score.rouge_1,
                "rouge_2": score.rouge_2,
                "rouge_l": score.rouge_l,
            }
        )

    average_scores = average_rouge_scores(scores)
    return {
        "rouge_1": average_scores.rouge_1,
        "rouge_2": average_scores.rouge_2,
        "rouge_l": average_scores.rouge_l,
        "predictions": predictions,
    }


__all__ = [
    "RougeScore",
    "average_rouge_scores",
    "evaluate_records",
    "evaluate_summary",
]
