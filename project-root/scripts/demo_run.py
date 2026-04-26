"""Demo run for extractive summarization models.

This script loads a few sample articles and generates summaries using all trained models.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.dataset_loader import (  # noqa: E402
    CNNDailyMailDatasetLoader,
    CNNDailyMailLoaderConfig,
)


def load_sample_articles(num_articles: int = 3) -> list[str]:
    """Load a few sample articles from the validation set."""
    loader = CNNDailyMailDatasetLoader(
        CNNDailyMailLoaderConfig(
            split="validation",
            sample_limit=num_articles,
            prefer_local_cache=True,
        )
    )
    records = loader.load_records()
    return [record.article_text for record in records]


def run_summarize(model_path: Path, text: str) -> str:
    """Run the summarize_article script and return the summary."""
    result = subprocess.run(
        [sys.executable, "scripts/summarize_article.py", "--model-path", str(model_path), "--text", text],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"Error: {result.stderr.strip()}"


def main() -> None:
    """Run the demo."""
    print("Loading sample articles...")
    articles = load_sample_articles(3)
    print(f"Loaded {len(articles)} articles.\n")

    models = {
        "logistic_regression": PROJECT_ROOT / "experiments" / "new_extractive_model.pkl",
        "linear_svm": PROJECT_ROOT / "experiments" / "new_linear_svm_model.pkl",
        "random_forest": PROJECT_ROOT / "experiments" / "new_random_forest_model.pkl",
        "mlp": PROJECT_ROOT / "experiments" / "new_mlp_model.pkl",
    }

    for i, article in enumerate(articles, 1):
        print(f"=== Article {i} ===")
        print(f"Original text (first 500 chars): {article[:500]}...\n")

        for model_name, model_path in models.items():
            if model_path.exists():
                print(f"--- {model_name.upper()} Summary ---")
                summary = run_summarize(model_path, article)
                print(summary)
            else:
                print(f"Warning: Model {model_name} not found at {model_path}")
            print()

        print("-" * 50)
        print()


if __name__ == "__main__":
    main()