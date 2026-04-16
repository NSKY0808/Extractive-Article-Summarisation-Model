# Multi-Document News Summarization System

This project implements a lightweight, explainable extractive summarization pipeline for news articles using classical machine learning. The system converts CNN/DailyMail article-summary pairs into sentence-level supervision with ROUGE-based heuristic labels, trains a sentence importance classifier, and generates summaries by ranking and selecting the best sentences.

## Pipeline

1. Load CNN/DailyMail article-summary pairs
2. Split each article into sentences
3. Generate pseudo-labels with ROUGE overlap against the reference summary
4. Extract TF-IDF and simple numeric sentence features
5. Train a binary sentence classifier
6. Rank sentences by predicted importance
7. Remove redundant sentences with cosine similarity filtering
8. Preserve original order and join top sentences into the final summary

## Project Structure

- `src/`
  - `dataset_loader.py`: dataset loading and sentence-level pseudo-label generation
  - `feature_pipeline.py`: TF-IDF and dense feature extraction
  - `summarizer.py`: inference-time ranking, redundancy removal, and summary generation
  - `evaluation.py`: ROUGE evaluation helpers
- `models/`
  - `extractive_classifier.py`: training, evaluation, save, and load wrapper for classical ML models
- `scripts/`
  - `prepare_dataset.py`: export sentence-level labels
  - `train_extractive_model.py`: train and validate a model
  - `evaluate_model.py`: evaluate a saved model on a dataset split
  - `summarize_article.py`: summarize a single article
- `experiments/`
  - stores trained models and metrics

## Requirements

Install the core dependencies:

```bash
python -m pip install datasets scikit-learn scipy numpy
```

## Training

Train the default logistic regression model:

```bash
python scripts/train_extractive_model.py --output-model-path experiments/extractive_model.pkl --metrics-output-path experiments/train_metrics.json
```

Train another supported model by passing `--model-type`:

```bash
python scripts/train_extractive_model.py --model-type linear_svm --output-model-path experiments/linear_svm_model.pkl --metrics-output-path experiments/linear_svm_train_metrics.json
python scripts/train_extractive_model.py --model-type random_forest --output-model-path experiments/random_forest_model.pkl --metrics-output-path experiments/random_forest_train_metrics.json
python scripts/train_extractive_model.py --model-type mlp --output-model-path experiments/mlp_model.pkl --metrics-output-path experiments/mlp_train_metrics.json
```

This command:

- loads `1000` training records from CNN/DailyMail
- loads `200` validation records
- converts them into sentence-level classification examples
- trains a logistic regression classifier
- saves the trained model to `experiments/extractive_model.pkl`
- saves validation metrics to `experiments/train_metrics.json`

## Evaluation

Evaluate the saved model on the test split:

```bash
python scripts/evaluate_model.py --model-path experiments/extractive_model.pkl --split test --sample-limit 200 --output-path experiments/test_metrics.json
```

Generate a summary for a single article:

```bash
python scripts/summarize_article.py --model-path experiments/extractive_model.pkl --text "Your news article here."
```

## Recorded Results

The following results were produced locally on April 15, 2026 using the current codebase.

Shared setup for all runs:

- Training records: `1000`
- Validation records: `200`
- Test records: `200`
- Training sentence examples: `28,529`
- Validation sentence examples: `5,229`

### Model Comparison

| Model | Validation Accuracy | Validation F1 | Validation ROUGE-1 | Validation ROUGE-2 | Validation ROUGE-L | Test ROUGE-1 | Test ROUGE-2 | Test ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logistic_regression` | `0.7085` | `0.4326` | `0.2407` | `0.0930` | `0.1667` | `0.2378` | `0.0845` | `0.1665` |
| `linear_svm` | `0.6927` | `0.3836` | `0.2409` | `0.0876` | `0.1638` | `0.2451` | `0.0860` | `0.1690` |
| `random_forest` | `0.8007` | `0.2794` | `0.2544` | `0.0982` | `0.1769` | `0.2600` | `0.1007` | `0.1842` |
| `mlp` | `0.7449` | `0.3038` | `0.2369` | `0.0793` | `0.1616` | `0.2379` | `0.0760` | `0.1642` |

### Best Result In This Run

- Best held-out summarization model: `random_forest`
- Test ROUGE-1: `0.2600`
- Test ROUGE-2: `0.1007`
- Test ROUGE-L: `0.1842`

## Saved Artifacts

- Model: [`experiments/extractive_model.pkl`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/extractive_model.pkl)
- Validation metrics: [`experiments/train_metrics.json`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/train_metrics.json)
- Test metrics: [`experiments/test_metrics.json`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/test_metrics.json)
- Linear SVM model and metrics: [`experiments/linear_svm_model.pkl`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/linear_svm_model.pkl), [`experiments/linear_svm_train_metrics.json`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/linear_svm_train_metrics.json), [`experiments/linear_svm_test_metrics.json`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/linear_svm_test_metrics.json)
- Random forest model and metrics: [`experiments/random_forest_model.pkl`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/random_forest_model.pkl), [`experiments/random_forest_train_metrics.json`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/random_forest_train_metrics.json), [`experiments/random_forest_test_metrics.json`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/random_forest_test_metrics.json)
- MLP model and metrics: [`experiments/mlp_model.pkl`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/mlp_model.pkl), [`experiments/mlp_train_metrics.json`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/mlp_train_metrics.json), [`experiments/mlp_test_metrics.json`](/c:/Users/prith/OneDrive/Desktop/BART-cnn-Article-Summarisation-Model/project-root/experiments/mlp_test_metrics.json)

## Notes

- The labels are heuristic, because CNN/DailyMail summaries are abstractive while the model is extractive.
- The current setup prioritizes explainability and CPU-friendly training over state-of-the-art summary quality.
- In this comparison, `random_forest` produced the strongest test ROUGE scores, while `linear_svm` remained competitive and `mlp` was the slowest model to train.
