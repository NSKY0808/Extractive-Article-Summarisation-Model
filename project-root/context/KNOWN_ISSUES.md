# KNOWN ISSUES

## Current Limitations

- Pseudo-labels are noisy because they are generated from ROUGE overlap, not human extractive annotations.
- The sentence tokenizer is regex-based and may split imperfectly on edge cases.
- Named entity counting is heuristic and based on capitalization patterns rather than a full NER system.
- Class imbalance is noticeable, so precision and recall can vary a lot across model types.
- Large local model artifacts such as random forest checkpoints can exceed GitHub's normal file size limits.

## Behavior Risks

- The classifier can select long but only partially relevant sentences.
- ROUGE can favor lexical overlap over concise summaries.
- Summaries may still include redundant details when similar sentences use different wording.

## Practical Guidance

- Keep metrics JSON files under `experiments/`, but avoid committing large `.pkl` model files.
- Use the README comparison table when choosing the default model.
- Treat `random_forest` as the current best summary model, not as a guaranteed best choice for every dataset size.