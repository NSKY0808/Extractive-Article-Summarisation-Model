# DATA PIPELINE

## Input Format

Cluster:

* List of articles (strings)

## Processing Steps

1. Sentence Tokenization
2. Remove duplicate sentences (cosine similarity > 0.8)
3. Rank sentences using TF-IDF or centroid similarity
4. Keep top 30% sentences

## Output Format

{
"input_text": "compressed cluster text"
}

## Rules

* Preserve factual sentences
* Prefer sentences with named entities
* Remove boilerplate news text

## Warning

Bad filtering = bad summaries
