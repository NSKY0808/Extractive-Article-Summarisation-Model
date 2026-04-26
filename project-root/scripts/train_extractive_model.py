"""Train a classical extractive summarization model on CNN/DailyMail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


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


def _parse_hidden_layer_sizes(value: str) -> tuple[int, ...]:
    """Parse a comma-separated hidden-layer specification."""

    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train a classical extractive summarizer.")
    parser.add_argument("--train-split", default="train", help="Training dataset split.")
    parser.add_argument("--validation-split", default="validation", help="Validation dataset split.")
    parser.add_argument("--train-limit", type=int, default=1500, help="Optional number of training records.")
    parser.add_argument("--validation-limit", type=int, default=200, help="Optional number of validation records.")
    parser.add_argument("--label-threshold", type=float, default=0.18, help="Threshold for positive labels.")
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=["logistic_regression", "linear_svm", "random_forest", "mlp"],
        help="Classifier family to train.",
    )
    parser.add_argument("--max-tfidf-features", type=int, default=8000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF features.")
    parser.add_argument("--max-df", type=float, default=0.9, help="Maximum document frequency for TF-IDF features.")
    parser.add_argument("--top-n-sentences", type=int, default=3, help="Number of sentences per generated summary.")
    parser.add_argument("--redundancy-threshold", type=float, default=0.8, help="Redundancy filter threshold.")
    parser.add_argument("--mmr-lambda", type=float, default=0.85, help="MMR relevance vs diversity balance.")
    parser.add_argument("--max-candidates", type=int, default=15, help="How many ranked sentences to consider.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees for random forest.")
    parser.add_argument("--logistic-c", type=float, default=2.0, help="Inverse regularization for logistic regression.")
    parser.add_argument("--svm-c", type=float, default=0.8, help="Inverse regularization for linear SVM.")
    parser.add_argument(
        "--hidden-layer-sizes",
        type=_parse_hidden_layer_sizes,
        default=(128, 64),
        help="Comma-separated hidden sizes for MLP, for example 128,64.",
    )
    parser.add_argument(
        "--prefer-local-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer the local Hugging Face cache and avoid network access when possible.",
    )
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


def load_split(split: str, sample_limit: int | None, prefer_local_cache: bool):
    """Load one dataset split from CNN/DailyMail."""

    loader = CNNDailyMailDatasetLoader(
        CNNDailyMailLoaderConfig(
            split=split,
            sample_limit=sample_limit,
            prefer_local_cache=prefer_local_cache,
        )
    )
    return loader.load_records()


def log_stage(message: str, start_time: float) -> None:
    """Print a timestamped training progress message."""

    elapsed_seconds = time.perf_counter() - start_time
    elapsed_minutes = elapsed_seconds / 60.0
    print(f"[{elapsed_minutes:7.2f} min] {message}", flush=True)


def main() -> None:
    """Train the extractive classifier and evaluate it on validation data."""

    args = parse_args()
    run_start = time.perf_counter()
    labeling_config = SentenceLabelingConfig(label_threshold=args.label_threshold)

    log_stage(
        f"Starting training for model_type={args.model_type}, train_limit={args.train_limit}, "
        f"validation_limit={args.validation_limit}",
        run_start,
    )

    train_records = load_split(args.train_split, args.train_limit, args.prefer_local_cache)
    log_stage(f"Loaded {len(train_records)} training records from split='{args.train_split}'", run_start)

    validation_records = load_split(args.validation_split, args.validation_limit, args.prefer_local_cache)
    log_stage(
        f"Loaded {len(validation_records)} validation records from split='{args.validation_split}'",
        run_start,
    )

    train_examples = build_sentence_classification_dataset(train_records, labeling_config)
    log_stage(f"Built {len(train_examples)} training sentence examples", run_start)

    validation_examples = build_sentence_classification_dataset(validation_records, labeling_config)
    log_stage(f"Built {len(validation_examples)} validation sentence examples", run_start)

    classifier = ExtractiveSentenceClassifier(
        model_config=ExtractiveModelConfig(
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            logistic_c=args.logistic_c,
            svm_c=args.svm_c,
            hidden_layer_sizes=args.hidden_layer_sizes,
        ),
        feature_config=SentenceFeatureConfig(
            max_tfidf_features=args.max_tfidf_features,
            min_df=args.min_df,
            max_df=args.max_df,
        ),
    )
    log_stage("Fitting classifier", run_start)
    classifier.fit(train_examples)
    log_stage("Finished classifier fit", run_start)

    sentence_metrics = classifier.evaluate(validation_examples)
    log_stage("Computed sentence-level validation metrics", run_start)

    summary_metrics = evaluate_records(
        validation_records,
        classifier,
        config=SummarizationConfig(
            top_n_sentences=args.top_n_sentences,
            redundancy_threshold=args.redundancy_threshold,
            mmr_lambda=args.mmr_lambda,
            max_candidates=args.max_candidates,
        ),
    )
    log_stage("Computed summary-level validation metrics", run_start)

    classifier.save(args.output_model_path)
    log_stage(f"Saved trained model to {args.output_model_path}", run_start)

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
        "model_type": args.model_type,
    }

    if args.metrics_output_path is not None:
        args.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(metrics_payload, file_handle, indent=2)
        log_stage(f"Saved metrics to {args.metrics_output_path}", run_start)

    print(json.dumps(metrics_payload, indent=2))
    print(f"Saved trained model to {args.output_model_path}")


if __name__ == "__main__":
    main()
