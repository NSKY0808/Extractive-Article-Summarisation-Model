"""Prepare a sentence-level classification dataset from CNN/DailyMail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.dataset_loader import (  # noqa: E402
    CNNDailyMailDatasetLoader,
    CNNDailyMailLoaderConfig,
    SentenceLabelingConfig,
    build_sentence_classification_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Prepare sentence-level extractive labels.")
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional maximum number of records.")
    parser.add_argument("--rouge-threshold", type=float, default=0.2, help="ROUGE threshold for positive labels.")
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to the output JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    """Load CNN/DailyMail and write sentence-level labeled examples to JSONL."""

    args = parse_args()
    loader = CNNDailyMailDatasetLoader(
        CNNDailyMailLoaderConfig(
            split=args.split,
            sample_limit=args.sample_limit,
        )
    )
    records = loader.load_records()
    examples = build_sentence_classification_dataset(
        records,
        config=SentenceLabelingConfig(rouge_threshold=args.rouge_threshold),
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as file_handle:
        for example in examples:
            file_handle.write(json.dumps(example.__dict__, ensure_ascii=True) + "\n")

    print(f"Wrote {len(examples)} sentence examples to {args.output_path}")


if __name__ == "__main__":
    main()
