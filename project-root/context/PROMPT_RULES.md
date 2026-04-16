# PROMPT RULES

## General Guidance

- Keep the repository aligned with the current extractive ML scope.
- Prefer changing documentation and configs to reflect the implemented system rather than outdated plans.
- Maintain consistency between `README.md`, `context/`, and the actual Python modules.

## When Writing Code

- Reuse the existing tokenizer, feature extractor, evaluator, and classifier wrapper where possible.
- Favor simple, inspectable heuristics over opaque complexity.
- Add new modules only when they have a clear pipeline responsibility.

## When Updating Docs

- Document the system as a classical extractive summarizer.
- Include concrete commands and measured metrics when available.
- Avoid references to BART, PEGASUS, or hierarchical abstractive pipelines unless the codebase actually supports them.

## When Unsure

- Make the smallest consistent change that keeps the repo coherent.
- Prefer documenting assumptions explicitly rather than leaving stale docs in place.