# PROJECT SPEC

## Project Name

Extractive Article Summarisation Model

## Objective

Build a lightweight supervised extractive summarization system for news articles that:

- uses CNN/DailyMail as the source dataset
- converts articles into sentence-level classification examples
- predicts sentence importance with classical machine learning
- generates summaries by selecting the best-ranked non-redundant sentences

## Scope

This repository currently implements single-article extractive summarization, not multi-document abstractive summarization.

## Constraints

- Must be explainable for academic evaluation
- Must run on CPU or modest hardware
- Must avoid transformer summarization models and LLM-based generation
- Must use modular code and reusable pipeline stages

## Success Criteria

- End-to-end training and inference scripts work
- Sentence-level classifier metrics are reported
- Summary-level ROUGE metrics are reported
- Documentation matches the implemented codebase

## Non-Goals

- Training a transformer from scratch
- Building a production-scale summarization service
- Multi-document fusion or meta-summary generation in the current version