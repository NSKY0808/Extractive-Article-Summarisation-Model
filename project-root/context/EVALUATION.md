# EVALUATION

## Primary Metrics

Sentence classification:

- Accuracy
- Precision
- Recall
- F1

Summarization quality:

- ROUGE-1
- ROUGE-2
- ROUGE-L

## Evaluation Procedure

1. Train on sentence-level pseudo-labels generated from CNN/DailyMail
2. Evaluate the classifier on validation sentence examples
3. Generate summaries on validation or test articles
4. Compare generated summaries to CNN/DailyMail highlights using ROUGE

## Current Benchmark Snapshot

Measured locally on April 15, 2026 with:

- train records: `1000`
- validation records: `200`
- test records: `200`

Best held-out summarization result in this repository:

- model: `random_forest`
- test ROUGE-1: `0.2600`
- test ROUGE-2: `0.1007`
- test ROUGE-L: `0.1842`

## Limitations

- The labels are heuristic because the reference summaries are abstractive.
- ROUGE rewards lexical overlap and does not fully capture summary usefulness.
- Sentence-level classifier quality and summary-level quality do not always move together.