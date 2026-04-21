# Extractive News Summarization System

This project implements a lightweight, explainable extractive summarization pipeline for news articles using classical machine learning. It converts CNN/DailyMail article-summary pairs into sentence-level supervision with ROUGE-based heuristic labels, trains a sentence importance classifier, and generates summaries by ranking and selecting the best sentences.

## What Improved

The latest benchmark run on April 20, 2026 focused on robustness and reproducibility, not just raw scores.

Key improvements in the current pipeline:

- offline-first CNN/DailyMail loading from the local Hugging Face cache
- stronger pseudo-labeling with weighted ROUGE-1, ROUGE-2, and ROUGE-L scoring
- removal of feature leakage from training-only label hints
- corrected sentence position features after filtering boilerplate sentences
- stronger TF-IDF plus dense sentence features
- improved MMR-based sentence selection with `mmr_lambda=0.85`
- larger training setup for the main benchmark: `1500` train records, `200` validation, `200` test

## Pipeline

1. Load CNN/DailyMail article-summary pairs
2. Split each article into sentences
3. Filter noisy or boilerplate sentences
4. Generate pseudo-labels with weighted ROUGE overlap against the reference summary
5. Extract TF-IDF and interpretable numeric sentence features
6. Train a binary sentence classifier
7. Rank sentences by predicted importance
8. Remove redundant sentences with cosine similarity filtering
9. Select the final summary with maximal marginal relevance and preserve original order

## Project Structure

- `project-root/src/`
  - `dataset_loader.py`: dataset loading and sentence-level pseudo-label generation
  - `feature_pipeline.py`: TF-IDF and dense feature extraction
  - `summarizer.py`: inference-time ranking, redundancy removal, and summary generation
  - `evaluation.py`: ROUGE evaluation helpers
- `project-root/models/`
  - `extractive_classifier.py`: training, evaluation, save, and load wrapper for classical ML models
- `project-root/scripts/`
  - `prepare_dataset.py`: export sentence-level labels
  - `train_extractive_model.py`: train and validate a model
  - `evaluate_model.py`: evaluate a saved model on a dataset split
  - `summarize_article.py`: summarize a single article
- `project-root/context/`
  - project notes, architecture, training, and evaluation context files
- `project-root/experiments/`
  - benchmark outputs and historical experiment artifacts

## Requirements

Install the core dependencies:

```bash
python -m pip install datasets scikit-learn scipy numpy
```

## Training

Train the default logistic regression model:

```bash
cd project-root
python scripts/train_extractive_model.py --output-model-path experiments/extractive_model.pkl --metrics-output-path experiments/train_metrics.json
```

Train another supported model by passing `--model-type`:

```bash
python scripts/train_extractive_model.py --model-type linear_svm --output-model-path experiments/linear_svm_model.pkl --metrics-output-path experiments/linear_svm_train_metrics.json
python scripts/train_extractive_model.py --model-type random_forest --output-model-path experiments/random_forest_model.pkl --metrics-output-path experiments/random_forest_train_metrics.json
python scripts/train_extractive_model.py --model-type mlp --output-model-path experiments/mlp_model.pkl --metrics-output-path experiments/mlp_train_metrics.json
```

Current default benchmark-oriented settings:

- training records: `1500`
- validation records: `200`
- TF-IDF vocabulary: `8000`
- top sentences per summary: `3`
- MMR lambda: `0.85`

## Evaluation

Evaluate a saved model on the test split:

```bash
python scripts/evaluate_model.py --model-path experiments/extractive_model.pkl --split test --sample-limit 200 --output-path experiments/test_metrics.json
```

Generate a summary for a single article:

```bash
python scripts/summarize_article.py --model-path experiments/extractive_model.pkl --text "Your news article here."
```

## Latest Results

Latest benchmark date: April 20, 2026.

Shared setup for all models:

- Train records: `1500`
- Validation records: `200`
- Test records: `200`
- Validation sentence examples: `5221`
- Summarization settings: `top_n=3`, `redundancy_threshold=0.8`, `mmr_lambda=0.85`, `max_candidates=15`

### Model Comparison

| Model | Validation Accuracy | Validation F1 | Validation ROUGE-1 | Validation ROUGE-2 | Validation ROUGE-L | Test ROUGE-1 | Test ROUGE-2 | Test ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logistic_regression` | `0.7895` | `0.3423` | `0.2517` | `0.0921` | `0.1734` | `0.2648` | `0.0950` | `0.1854` |
| `linear_svm` | `0.8914` | `0.1014` | `0.2492` | `0.0863` | `0.1692` | `0.2612` | `0.0911` | `0.1823` |
| `random_forest` | `0.8431` | `0.3947` | `0.2670` | `0.1020` | `0.1865` | `0.2799` | `0.1064` | `0.1946` |
| `mlp` | `0.8937` | `0.3305` | `0.2688` | `0.1008` | `0.1844` | `0.2735` | `0.1032` | `0.1909` |

### Best Model

- Best held-out summarization model: `random_forest`
- Test ROUGE-1: `0.2799`
- Test ROUGE-2: `0.1064`
- Test ROUGE-L: `0.1946`

### Improvement Over The Previous Benchmark

Compared with the earlier April 15, 2026 run, every model improved on held-out test ROUGE-1:

- `logistic_regression`: `0.2378 -> 0.2648` (`+0.0270`)
- `linear_svm`: `0.2451 -> 0.2612` (`+0.0161`)
- `random_forest`: `0.2600 -> 0.2799` (`+0.0199`)
- `mlp`: `0.2379 -> 0.2735` (`+0.0356`)

## Benchmark Artifact

The latest evaluation summary is stored in:

- `project-root/experiments/improved/benchmark_summary.json`

This file records the April 20, 2026 benchmark metrics used in this README.

## Notes

- Labels are heuristic because CNN/DailyMail summaries are abstractive while the model is extractive.
- The project prioritizes interpretability, CPU-friendly training, and viva-friendly explainability over state-of-the-art neural summarization.
- `random_forest` currently gives the strongest end-to-end summary quality, while `logistic_regression` remains a strong lightweight baseline.
- `linear_svm` achieves high sentence classification accuracy but is conservative about positive predictions, so sentence-level F1 and summary quality do not move together perfectly.
