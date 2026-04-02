# PROJECT SPEC — Multi-Document News Summarization

## Objective

Build an abstractive summarization system that:

* Takes clustered news articles (same event)
* Produces a coherent, non-redundant summary

## Constraints

* Input may exceed model token limits
* Articles contain redundancy and noise
* No human-labeled summaries available

## Current Approach

* Use pretrained BART model
* Generate pseudo-label summaries
* Fine-tune model on clustered dataset

## Success Criteria

* High ROUGE-L score on evaluation set
* Low redundancy in summaries
* Minimal hallucination

## Non-Goals

* Not building a general-purpose LLM
* Not training transformer from scratch

## System Type

* Hybrid pipeline (extractive + abstractive)
