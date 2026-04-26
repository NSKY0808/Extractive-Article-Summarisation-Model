#!/bin/bash
# Train all 4 models on 15k dataset

cd "$(dirname "$0")/.."

echo "Starting training of all 4 models on 15k dataset..."
echo "============================================"

echo ""
echo "Training Logistic Regression..."
python scripts/train_extractive_model.py --model-type logistic_regression --train-limit 15000 --validation-limit 2000 --output-model-path experiments/logistic_regression_15k_model.pkl --max-tfidf-features 8000

echo ""
echo "Training Linear SVM..."
python scripts/train_extractive_model.py --model-type linear_svm --train-limit 15000 --validation-limit 2000 --output-model-path experiments/linear_svm_15k_model.pkl --max-tfidf-features 8000

echo ""
echo "Training Random Forest..."
python scripts/train_extractive_model.py --model-type random_forest --train-limit 15000 --validation-limit 2000 --output-model-path experiments/random_forest_15k_model.pkl --max-tfidf-features 8000

echo ""
echo "Training MLP..."
python scripts/train_extractive_model.py --model-type mlp --train-limit 15000 --validation-limit 2000 --output-model-path experiments/mlp_15k_model.pkl --max-tfidf-features 8000

echo ""
echo "============================================"
echo "All training complete!"
