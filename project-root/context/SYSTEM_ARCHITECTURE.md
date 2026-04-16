# SYSTEM ARCHITECTURE

## High-Level Flow

1. Dataset loading
2. Sentence tokenization
3. ROUGE-based pseudo-label generation
4. Feature extraction
5. Sentence importance classification
6. Sentence ranking
7. Redundancy removal
8. Summary assembly
9. ROUGE evaluation

## Module Layout

- `src/dataset_loader.py` -> dataset access and pseudo-labeling
- `src/data_pipeline.py` -> tokenization and low-level text utilities
- `src/feature_pipeline.py` -> TF-IDF and dense sentence features
- `models/extractive_classifier.py` -> model training, scoring, save, and load
- `src/summarizer.py` -> ranking and summary generation
- `src/evaluation.py` -> summary scoring
- `scripts/*.py` -> command-line entrypoints

## Data Flow

`article + reference summary`
-> `sentence examples`
-> `feature matrix`
-> `classifier`
-> `sentence scores`
-> `redundancy filter`
-> `extractive summary`

## Design Principles

- Keep training and inference paths separate but compatible.
- Preserve narrative order after ranking.
- Use sparse TF-IDF plus simple dense features for interpretability.