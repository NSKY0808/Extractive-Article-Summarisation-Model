# CODING RULES

## Language

Python

## Project Style

- Keep the system modular and pipeline-based.
- Separate data loading, feature engineering, model logic, inference, and evaluation.
- Prefer classical machine learning and small, explainable heuristics over heavyweight models.

## Folder Responsibilities

- `context/` -> project definitions, constraints, and documentation
- `src/` -> reusable pipeline logic
- `models/` -> trainable classifier wrappers
- `scripts/` -> runnable entrypoints
- `configs/` -> JSON configuration examples and defaults
- `data/` -> local exports, prepared datasets, and notes about dataset layout
- `experiments/` -> metrics and local model artifacts

## Rules

- Do not hardcode dataset paths or machine-specific paths in the Python code.
- Keep functions small and reusable.
- Add docstrings to public functions and classes.
- Prefer explicit dataclasses and configuration objects for pipeline settings.
- Do not reintroduce transformer summarization code into this repository unless the project scope changes.