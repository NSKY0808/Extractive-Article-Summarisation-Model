# TRAINING PIPELINE

## Dataset Creation

For each CNN/DailyMail record:

1. load `article`
2. load `highlights`
3. tokenize the article into sentences
4. compute ROUGE-based overlap between each sentence and the reference summary
5. assign a binary label using the configured threshold

## Feature Creation

Combine:

- TF-IDF sentence vectors
- sentence position features
- sentence length features
- simple entity and number heuristics

## Model Training Steps

1. build sentence-level examples
2. fit the shared feature extractor
3. train the selected classifier
4. evaluate sentence classification metrics on validation examples
5. generate validation summaries
6. evaluate ROUGE on the generated summaries

## Default Run Used In This Repo

- model: `logistic_regression`
- train records: `1000`
- validation records: `200`
- test records: `200`

## Supported Training Command

```bash
python scripts/train_extractive_model.py --model-type logistic_regression --output-model-path experiments/extractive_model.pkl --metrics-output-path experiments/train_metrics.json
```

## Rules

- Keep validation and test splits separate from training.
- Compare model families with the same feature pipeline and sample sizes.
- Save summary metrics as JSON for reproducibility.