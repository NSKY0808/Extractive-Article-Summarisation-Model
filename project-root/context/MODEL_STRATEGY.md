# MODEL STRATEGY

## Model Used

facebook/bart-large-cnn

## Why This Model

* Pretrained for summarization
* Handles noisy input well
* Stable fine-tuning

## Input Strategy

* Max token limit: 1024
* Use chunking for long inputs

## Output Strategy

* Max summary length: 150 tokens

## Future Upgrades

* PEGASUS for better quality
* Longformer for long input handling
