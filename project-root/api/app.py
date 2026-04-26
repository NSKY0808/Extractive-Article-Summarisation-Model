"""Flask API for the extractive summarization demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.extractive_classifier import ExtractiveSentenceClassifier  # noqa: E402
from src.summarizer import SummarizationConfig, summarize_article  # noqa: E402

app = Flask(__name__)
CORS(app)

MODELS = {}
MODEL_PATHS = {
    "logistic_regression": "experiments/logistic_regression_15k_model.pkl",
    "linear_svm": "experiments/linear_svm_15k_model.pkl",
    "random_forest": "experiments/random_forest_15k_model.pkl",
    "mlp": "experiments/mlp_15k_model.pkl",
}
BENCHMARK_SUMMARY_PATH = PROJECT_ROOT / "experiments" / "improved" / "benchmark_summary.json"


def load_benchmark_metrics() -> Dict[str, object]:
    """Load the benchmark summary used by the README and demo frontend."""

    if not BENCHMARK_SUMMARY_PATH.exists():
        return {}

    with BENCHMARK_SUMMARY_PATH.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_model(model_name: str) -> ExtractiveSentenceClassifier | None:
    """Load a model from disk, with caching."""

    if model_name in MODELS:
        return MODELS[model_name]

    model_path = PROJECT_ROOT / MODEL_PATHS[model_name]
    if not model_path.exists():
        return None

    try:
        model = ExtractiveSentenceClassifier.load(model_path)
        MODELS[model_name] = model
        return model
    except Exception as error:
        print(f"Error loading model {model_name}: {error}")
        return None


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""

    return jsonify({"status": "ok"})


@app.route("/api/models", methods=["GET"])
def list_models():
    """List available models and their status."""

    models_info = {}
    for model_name, model_path in MODEL_PATHS.items():
        path = PROJECT_ROOT / model_path
        models_info[model_name] = {
            "available": path.exists(),
            "path": str(path),
        }
    return jsonify(models_info)


@app.route("/api/benchmark-metrics", methods=["GET"])
def benchmark_metrics():
    """Return benchmark metrics for the four demo models."""

    benchmark_data = load_benchmark_metrics()
    if not benchmark_data:
        return jsonify({"error": f"Benchmark summary not found at {BENCHMARK_SUMMARY_PATH}"}), 404
    return jsonify(benchmark_data)


@app.route("/api/summarize", methods=["POST"])
def summarize():
    """Generate summaries for the input article using all available models."""

    data = request.get_json()

    if not data or "article" not in data:
        return jsonify({"error": "Missing 'article' in request body"}), 400

    article_text = data["article"].strip()
    if not article_text:
        return jsonify({"error": "Article cannot be empty"}), 400

    config = SummarizationConfig(
        top_n_sentences=data.get("top_n_sentences", 3),
        redundancy_threshold=data.get("redundancy_threshold", 0.8),
        mmr_lambda=data.get("mmr_lambda", 0.85),
        max_candidates=data.get("max_candidates", 15),
    )

    results = {
        "article": article_text,
        "summaries": {},
        "errors": {},
    }

    for model_name in MODEL_PATHS:
        try:
            model = load_model(model_name)
            if model is None:
                results["errors"][model_name] = f"Model not found at {MODEL_PATHS[model_name]}"
                continue

            summary_result = summarize_article(article_text, model, config)
            results["summaries"][model_name] = {
                "summary": summary_result["summary_text"],
                "sentences": summary_result.get("selected_sentence_indices", []),
            }
        except Exception as error:
            results["errors"][model_name] = str(error)

    return jsonify(results)


@app.route("/api/compare", methods=["POST"])
def compare():
    """Generate summaries and comparison metrics."""

    data = request.get_json()

    if not data or "article" not in data:
        return jsonify({"error": "Missing 'article' in request body"}), 400

    article_text = data["article"].strip()
    if not article_text:
        return jsonify({"error": "Article cannot be empty"}), 400

    summaries = {}
    for model_name in MODEL_PATHS:
        try:
            model = load_model(model_name)
            if model is None:
                continue

            result = summarize_article(article_text, model, SummarizationConfig())
            summaries[model_name] = result["summary_text"]
        except Exception as error:
            print(f"Error summarizing with {model_name}: {error}")

    summary_lengths = {model: len(summary.split()) for model, summary in summaries.items()}

    return jsonify(
        {
            "article": article_text[:500] + "..." if len(article_text) > 500 else article_text,
            "summaries": summaries,
            "metrics": {
                "summary_lengths": summary_lengths,
                "article_length": len(article_text.split()),
            },
        }
    )


if __name__ == "__main__":
    print("Starting Extractive Summarization API...")
    print(f"Project root: {PROJECT_ROOT}")
    print("Available models:")
    for model_name, model_path in MODEL_PATHS.items():
        path = PROJECT_ROOT / model_path
        status = "[OK] Found" if path.exists() else "[MISSING] Not found"
        print(f"  - {model_name}: {status}")

    app.run(debug=True, host="0.0.0.0", port=5000)
