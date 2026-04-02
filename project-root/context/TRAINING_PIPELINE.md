# TRAINING PIPELINE

## Dataset Creation

For each cluster:

* Generate pseudo-summary using pretrained BART
* Store (input_text, target_text)

## Training Steps

1. Tokenize input and output
2. Fine-tune model using cross-entropy loss
3. Evaluate after each epoch

## Hyperparameters

learning_rate = 3e-5
batch_size = 2
epochs = 3–5

## Rules

* Do NOT train from scratch
* Always validate on separate dataset
* Avoid overfitting to pseudo-labels
