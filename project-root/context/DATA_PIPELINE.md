# DATA PIPELINE

## Input Format

Article-summary pairs from the CNN/DailyMail dataset.

- Input article: full article text
- Reference summary: abstractive highlights field from the dataset

## Processing Steps

1. Load records with `CNNDailyMailDatasetLoader`
2. Split article text into sentences with the lightweight regex tokenizer
3. Remove very short or low-value sentences during downstream processing
4. Generate sentence-level pseudo-labels using ROUGE overlap against the reference summary
5. Extract sentence features:
   - TF-IDF
   - sentence position
   - reverse position
   - sentence length
   - named-entity-count heuristic
   - number-presence flag

## Training Output

Each sentence becomes a binary classification example with:

- `sentence_text`
- `sentence_index`
- `sentence_count`
- `label`
- `rouge1_f1`
- `rouge_l_f1`

## Inference Output

The summarizer returns:

- ranked sentence scores
- selected sentences after redundancy filtering
- final `summary_text`

## Rules

- Do not feed raw article text directly into a summarization model.
- Preserve original sentence order in the final extractive summary.
- Remove redundant selected sentences with cosine similarity filtering.
- Keep the pipeline explainable for course presentation and viva discussion.