"""Evaluate a trained extractive summarization model on CNN/DailyMail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.extractive_classifier import ExtractiveSentenceClassifier  # noqa: E402
from src.dataset_loader import CNNDailyMailDatasetLoader, CNNDailyMailLoaderConfig  # noqa: E402
from src.evaluation import evaluate_records  # noqa: E402
from src.summarizer import SummarizationConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate a trained extractive summarizer.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a saved model bundle.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--sample-limit", type=int, default=200, help="Optional number of records to evaluate.")
    parser.add_argument("--top-n-sentences", type=int, default=3, help="Number of sentences to include in summaries.")
    parser.add_argument("--redundancy-threshold", type=float, default=0.8, help="Redundancy filter threshold.")
    parser.add_argument("--mmr-lambda", type=float, default=0.85, help="MMR relevance vs diversity balance.")
    parser.add_argument("--max-candidates", type=int, default=15, help="How many ranked sentences to consider.")
    parser.add_argument(
        "--prefer-local-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer the local Hugging Face cache and avoid network access when possible.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to save evaluation details as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a trained model and evaluate it on one dataset split."""

    args = parse_args()
    classifier = ExtractiveSentenceClassifier.load(args.model_path)
    loader = CNNDailyMailDatasetLoader(
        CNNDailyMailLoaderConfig(
            split=args.split,
            sample_limit=args.sample_limit,
            prefer_local_cache=args.prefer_local_cache,
        )
    )
    records = loader.load_records()
    metrics = evaluate_records(
        records,
        classifier,
        config=SummarizationConfig(
            top_n_sentences=args.top_n_sentences,
            redundancy_threshold=args.redundancy_threshold,
            mmr_lambda=args.mmr_lambda,
            max_candidates=args.max_candidates,
        ),
    )

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(metrics, file_handle, indent=2)

    printable_metrics = {
        "rouge_1": metrics["rouge_1"],
        "rouge_2": metrics["rouge_2"],
        "rouge_l": metrics["rouge_l"],
        "evaluated_records": len(metrics["predictions"]),
    }
    print(json.dumps(printable_metrics, indent=2))


if __name__ == "__main__":
    main()