# Data Folder
Use this folder for local data artifacts that are safe to regenerate or keep outside git history.
Recommended contents:
- exported JSONL sentence-label datasets from scripts/prepare_dataset.py
- small hand-curated evaluation samples
- notes about any custom preprocessing experiments
Do not store the full downloaded CNN/DailyMail dataset here unless you explicitly want a local copy.