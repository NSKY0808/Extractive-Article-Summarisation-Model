"""Train a classical extractive summarization model on CNN/DailyMail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.extractive_classifier import ExtractiveModelConfig, ExtractiveSentenceClassifier  # noqa: E402
from src.dataset_loader import (  # noqa: E402
    CNNDailyMailDatasetLoader,
    CNNDailyMailLoaderConfig,
    SentenceLabelingConfig,
    build_sentence_classification_dataset,
)
from src.evaluation import evaluate_records  # noqa: E402
from src.feature_pipeline import SentenceFeatureConfig  # noqa: E402
from src.summarizer import SummarizationConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train a classical extractive summarizer.")
    parser.add_argument("--train-split", default="train", help="Training dataset split.")
    parser.add_argument("--validation-split", default="validation", help="Validation dataset split.")
    parser.add_argument("--train-limit", type=int, default=1000, help="Optional number of training records.")
    parser.add_argument("--validation-limit", type=int, default=200, help="Optional number of validation records.")
    parser.add_argument("--rouge-threshold", type=float, default=0.2, help="Threshold for positive labels.")
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=["logistic_regression", "linear_svm", "random_forest", "mlp"],
        help="Classifier family to train.",
    )
    parser.add_argument("--max-tfidf-features", type=int, default=5000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF features.")
    parser.add_argument("--top-n-sentences", type=int, default=3, help="Number of sentences per generated summary.")
    parser.add_argument(
        "--output-model-path",
        type=Path,
        required=True,
        help="Path to save the trained model bundle.",
    )
    parser.add_argument(
        "--metrics-output-path",
        type=Path,
        default=None,
        help="Optional path to save evaluation metrics as JSON.",
    )
    return parser.parse_args()


def load_split(split: str, sample_limit: int | None):
    """Load one dataset split from CNN/DailyMail."""

    loader = CNNDailyMailDatasetLoader(
        CNNDailyMailLoaderConfig(
            split=split,
            sample_limit=sample_limit,
        )
    )
    return loader.load_records()


def main() -> None:
    """Train the extractive classifier and evaluate it on validation data."""

    args = parse_args()
    labeling_config = SentenceLabelingConfig(rouge_threshold=args.rouge_threshold)

    train_records = load_split(args.train_split, args.train_limit)
    validation_records = load_split(args.validation_split, args.validation_limit)

    train_examples = build_sentence_classification_dataset(train_records, labeling_config)
    validation_examples = build_sentence_classification_dataset(validation_records, labeling_config)

    classifier = ExtractiveSentenceClassifier(
        model_config=ExtractiveModelConfig(model_type=args.model_type),
        feature_config=SentenceFeatureConfig(
            max_tfidf_features=args.max_tfidf_features,
            min_df=args.min_df,
        ),
    )
    classifier.fit(train_examples)

    sentence_metrics = classifier.evaluate(validation_examples)
    summary_metrics = evaluate_records(
        validation_records,
        classifier,
        config=SummarizationConfig(top_n_sentences=args.top_n_sentences),
    )

    classifier.save(args.output_model_path)

    metrics_payload = {
        "sentence_classification": sentence_metrics,
        "summarization": {
            "rouge_1": summary_metrics["rouge_1"],
            "rouge_2": summary_metrics["rouge_2"],
            "rouge_l": summary_metrics["rouge_l"],
        },
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
    }

    if args.metrics_output_path is not None:
        args.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(metrics_payload, file_handle, indent=2)

    print(json.dumps(metrics_payload, indent=2))
    print(f"Saved trained model to {args.output_model_path}")


if __name__ == "__main__":
    main()
