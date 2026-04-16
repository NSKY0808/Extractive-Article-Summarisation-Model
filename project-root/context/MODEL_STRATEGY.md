# MODEL STRATEGY

## Supported Models

- `logistic_regression`
- `linear_svm`
- `random_forest`
- `mlp`

## Selection Principles

- Prefer explainable, CPU-friendly models.
- Keep training feasible on modest hardware.
- Use the same feature pipeline across models for fair comparison.

## Current Recommendation

Default teaching baseline:

- `logistic_regression`

Best summarization result in current experiments:

- `random_forest`

## Feature Strategy

All models use the same sentence representation:

- TF-IDF sentence vectors
- relative position
- reverse position
- token count
- normalized sentence length
- named-entity-count heuristic
- number-presence flag

## Explicit Non-Goals

- No large language models
- No pretrained abstractive summarizers
- No transformer fine-tuning inside this repository