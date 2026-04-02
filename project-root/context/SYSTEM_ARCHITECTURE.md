# SYSTEM ARCHITECTURE

## Pipeline Overview

1. Article Fetching
2. Clustering by Event
3. Sentence Filtering (extractive layer)
4. Chunking (token limit handling)
5. Abstractive Summarization (BART)
6. Meta-Summary Generation

## Key Principle

DO NOT pass raw articles directly into model.

## Design Decisions

* Use hierarchical summarization
* Use extractive filtering before transformer
* Use pseudo-labeling for training

## Data Flow

Cluster → Filter → Chunk → Summarize → Merge → Final Summary
