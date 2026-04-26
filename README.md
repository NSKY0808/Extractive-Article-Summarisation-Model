# Extractive News Summarization System

This project implements a lightweight, explainable extractive summarization pipeline for news articles using classical machine learning. It converts CNN/DailyMail article-summary pairs into sentence-level supervision with ROUGE-based heuristic labels, trains a sentence importance classifier, and generates summaries by ranking and selecting the best sentences.

## Current Status: Production Models Ready ✅

All four classical ML models have been successfully trained on the 15,000 record CNN/DailyMail dataset and are deployed:

- **Logistic Regression**: ROUGE-1 0.288 | ROUGE-2 0.111 | ROUGE-L 0.191
- **Linear SVM**: ROUGE-1 0.289 | ROUGE-2 0.111 | ROUGE-L 0.192
- **Random Forest**: ROUGE-1 0.307 | ROUGE-2 0.119 | ROUGE-L 0.203 (Best performer)
- **MLP**: ROUGE-1 0.296 | ROUGE-2 0.114 | ROUGE-L 0.195

The production models are located in `experiments/` and automatically loaded by the Flask API and frontend:
- `logistic_regression_15k_model.pkl`
- `linear_svm_15k_model.pkl`
- `random_forest_15k_model.pkl`
- `mlp_15k_model.pkl`

## Pipeline Features

The latest benchmark on April 27, 2026 demonstrates robust performance across all models:

- offline-first CNN/DailyMail loading from the local Hugging Face cache
- stronger pseudo-labeling with weighted ROUGE-1, ROUGE-2, and ROUGE-L scoring
- removal of feature leakage from training-only label hints
- corrected sentence position features after filtering boilerplate sentences
- stronger TF-IDF plus dense sentence features
- improved MMR-based sentence selection with `mmr_lambda=0.85`
- large-scale training setup: `15,000` train records, `2,000` validation, full dataset test split

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

The models have been trained on the 15,000 record dataset. All four production models are ready to use:

**To use the pre-trained 15k models**, simply run the frontend and API:

```bash
# Start the full application (both API and frontend)
# Windows
start.bat
# or Linux/Mac
./start.sh
```

The Flask API will automatically load the 15k models and serve them via the React frontend.

### Re-training on 15k Dataset (Optional)

To retrain all four 15k models on Windows:

```bat
cd project-root
scripts\train_all_15k.bat
```

To train individual models:

```bash
cd project-root
python scripts/train_extractive_model.py --model-type logistic_regression --train-limit 15000 --validation-limit 2000 --output-model-path experiments/logistic_regression_15k_model.pkl --max-tfidf-features 8000
python scripts/train_extractive_model.py --model-type linear_svm --train-limit 15000 --validation-limit 2000 --output-model-path experiments/linear_svm_15k_model.pkl --max-tfidf-features 8000
python scripts/train_extractive_model.py --model-type random_forest --train-limit 15000 --validation-limit 2000 --output-model-path experiments/random_forest_15k_model.pkl --max-tfidf-features 8000
python scripts/train_extractive_model.py --model-type mlp --train-limit 15000 --validation-limit 2000 --output-model-path experiments/mlp_15k_model.pkl --max-tfidf-features 8000
```

**Estimated training time:**
- Logistic Regression: 2-3 min
- Linear SVM: 2-3 min  
- Random Forest: 60-90 min
- MLP: 120-180 min
- **Total: ~4-8 hours**

Training all models with the batch script will use `project-root\api\venv\Scripts\python.exe` and output to:
- `experiments\logistic_regression_15k_model.pkl`
- `experiments\linear_svm_15k_model.pkl`
- `experiments\random_forest_15k_model.pkl`
- `experiments\mlp_15k_model.pkl`

Plus matching `*_15k_metrics.json` files for each model.

Current default settings:

- training records: `15,000`
- validation records: `2,000`
- TF-IDF vocabulary: `8,000`
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

## Latest Results - 15k Dataset

Latest benchmark date: April 27, 2026.

All four models have been trained and evaluated on the 15,000 record CNN/DailyMail dataset. These metrics are sourced from individual model evaluation runs and represent the current production state.

Shared setup for all models:

- Train records: `15,000`
- Validation records: `2,000`
- Summarization settings: `top_n=3`, `redundancy_threshold=0.8`, `mmr_lambda=0.85`, `max_candidates=15`

### Model Comparison (15k Training Dataset)

| Model | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 | Validation ROUGE-1 | Validation ROUGE-2 | Validation ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logistic_regression` | `0.7624` | `0.2957` | `0.6539` | `0.4072` | `0.2881` | `0.1109` | `0.1912` |
| `linear_svm` | `0.8775` | `0.5324` | `0.1491` | `0.2330` | `0.2888` | `0.1115` | `0.1916` |
| `random_forest` | `0.8419` | `0.3938` | `0.4954` | `0.4388` | `0.3075` | `0.1190` | `0.2030` |
| `mlp` | `0.8835` | `0.6139` | `0.1785` | `0.2766` | `0.2958` | `0.1142` | `0.1952` |

### Best Model (15k Dataset)

- Best summarization model: `random_forest`
- ROUGE-1: `0.3075` (+0.0277 vs benchmark)
- ROUGE-2: `0.1190` (+0.0126 vs benchmark)
- ROUGE-L: `0.2030` (+0.0084 vs benchmark)

### Improvement Over Previous Benchmark (1.5k → 15k Dataset)

All models show improved or maintained performance when trained on 10x more data:

- `logistic_regression`: `0.2648 -> 0.2881` (+0.0233)
- `linear_svm`: `0.2612 -> 0.2888` (+0.0276)
- `random_forest`: `0.2799 -> 0.3075` (+0.0276) **[Largest gain]**
- `mlp`: `0.2735 -> 0.2958` (+0.0223)

## Demo Application

The demo frontend displays summaries generated by all four models in real-time:

1. **Demo Tab**: Paste an article and generate side-by-side summaries from all four models
2. **Comparison Tab**: View detailed metrics, charts, and performance comparisons

The frontend connects to the Flask API which automatically loads the 15k production models from `project-root/experiments/` and serves them via REST endpoints:
- `/api/summarize` - Generate summaries for an article
- `/api/benchmark-metrics` - Retrieve benchmark metrics
- `/api/models` - Check model availability
- `/api/health` - Health check

To run the complete demo:

```bash
# Windows
start.bat
# Linux/Mac
./start.sh
```

This starts the Flask API on port 5000 and the React frontend on port 3000. Open http://localhost:3000 in your browser.

## Benchmark Data

The model evaluation metrics are stored in individual JSON files:

- `project-root/experiments/logistic_regression_15k_metrics.json`
- `project-root/experiments/linear_svm_15k_metrics.json`
- `project-root/experiments/random_forest_15k_metrics.json`
- `project-root/experiments/mlp_15k_metrics.json`
- `project-root/experiments/improved/benchmark_summary.json` (aggregated metrics for frontend display)

## Notes

- Labels are heuristic because CNN/DailyMail summaries are abstractive while the model is extractive.
- The project prioritizes interpretability, CPU-friendly training, and viva-friendly explainability over state-of-the-art neural summarization.
- `random_forest` currently gives the strongest end-to-end summary quality, while `logistic_regression` remains a strong lightweight baseline.
- `linear_svm` achieves high sentence classification accuracy but is conservative about positive predictions, so sentence-level F1 and summary quality do not move together perfectly.
