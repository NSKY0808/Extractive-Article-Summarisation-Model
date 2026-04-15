"""Generate an extractive summary for a single article."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.extractive_classifier import ExtractiveSentenceClassifier  # noqa: E402
from src.summarizer import SummarizationConfig, summarize_article  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate an extractive summary for one article.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a saved model bundle.")
    parser.add_argument("--text", default=None, help="Raw article text.")
    parser.add_argument("--input-file", type=Path, default=None, help="Path to a file containing article text.")
    parser.add_argument("--top-n-sentences", type=int, default=3, help="Number of sentences in the summary.")
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for duplicate removal.",
    )
    return parser.parse_args()


def load_article_text(args: argparse.Namespace) -> str:
    """Load article text from a direct argument or an input file."""

    if args.text:
        return args.text.strip()
    if args.input_file:
        return args.input_file.read_text(encoding="utf-8").strip()
    raise ValueError("Either --text or --input-file must be provided.")


def main() -> None:
    """Load a trained model and summarize a single article."""

    args = parse_args()
    article_text = load_article_text(args)
    classifier = ExtractiveSentenceClassifier.load(args.model_path)
    prediction = summarize_article(
        article_text,
        classifier,
        config=SummarizationConfig(
            top_n_sentences=args.top_n_sentences,
            redundancy_threshold=args.redundancy_threshold,
        ),
    )
    print(prediction["summary_text"])


if __name__ == "__main__":
    main()
