# Complete Setup Guide - Extractive Summarization System

End-to-end setup guide for training models on 15k dataset and running the full demo with React frontend.

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│           React Frontend (port 3000)                    │
├─────────────────────────────────────────────────────────┤
│  - Article Input Interface                              │
│  - Summary Display (4 models side-by-side)              │
│  - Comparison Dashboard with charts & metrics           │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP/JSON
                         │ (proxy: localhost:5000)
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Flask API (port 5000)                         │
├─────────────────────────────────────────────────────────┤
│  - Model Loading & Caching                              │
│  - Summarization Endpoints                              │
│  - Comparison Metrics Calculation                       │
└────────────────────────┬────────────────────────────────┘
                         │ Python/Pickle
                         │ (model loading)
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Trained Models (15k dataset)                  │
├─────────────────────────────────────────────────────────┤
│  - logistic_regression_15k_model.pkl                    │
│  - linear_svm_15k_model.pkl                             │
│  - random_forest_15k_model.pkl                          │
│  - mlp_15k_model.pkl                                    │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.8+ (for backend)
- Node.js 14+ (for frontend)
- npm or yarn
- 8GB+ RAM (for training on 15k dataset)
- 50GB+ disk space (for dataset cache and models)

## Step 1: Train Models on 15k Dataset

### Option A: Train All Models Sequentially (Recommended)

```bash
cd project-root

# Train Logistic Regression (~2-3 minutes)
python scripts/train_extractive_model.py \
  --model-type logistic_regression \
  --train-limit 15000 \
  --validation-limit 2000 \
  --output-model-path experiments/logistic_regression_15k_model.pkl \
  --max-tfidf-features 8000

# Train Linear SVM (~2-3 minutes)
python scripts/train_extractive_model.py \
  --model-type linear_svm \
  --train-limit 15000 \
  --validation-limit 2000 \
  --output-model-path experiments/linear_svm_15k_model.pkl \
  --max-tfidf-features 8000

# Train Random Forest (~60-90 minutes)
python scripts/train_extractive_model.py \
  --model-type random_forest \
  --train-limit 15000 \
  --validation-limit 2000 \
  --output-model-path experiments/random_forest_15k_model.pkl \
  --max-tfidf-features 8000

# Train MLP (~120-180 minutes)
python scripts/train_extractive_model.py \
  --model-type mlp \
  --train-limit 15000 \
  --validation-limit 2000 \
  --output-model-path experiments/mlp_15k_model.pkl \
  --max-tfidf-features 8000
```

**Total training time: ~4-8 hours** (depending on hardware)

### Option B: Using Shell Script (Linux/Mac)

```bash
cd project-root
chmod +x scripts/train_all_15k.sh
./scripts/train_all_15k.sh
```

### Option C: Train in Background (Windows PowerShell)

```powershell
cd project-root
Start-Process python -ArgumentList "scripts/train_extractive_model.py --model-type logistic_regression --train-limit 15000 --validation-limit 2000 --output-model-path experiments/logistic_regression_15k_model.pkl --max-tfidf-features 8000" -NoNewWindow
```

**Training Output:**
Each model saves:
- Model file: `experiments/{model_name}_15k_model.pkl`
- Metrics file: Contains ROUGE scores and accuracy metrics

## Step 2: Setup Flask Backend API

### 2.1 Install Backend Dependencies

```bash
cd api
pip install -r requirements.txt
```

### 2.2 Verify Models are Ready

```bash
python -c "
import os
models = [
    'logistic_regression_15k_model.pkl',
    'linear_svm_15k_model.pkl',
    'random_forest_15k_model.pkl',
    'mlp_15k_model.pkl'
]
for m in models:
    path = f'../experiments/{m}'
    print(f'{m}: {'✓ Found' if os.path.exists(path) else '✗ Not found'}')
"
```

### 2.3 Start the Flask API

```bash
python app.py
```

Expected output:
```
Starting Extractive Summarization API...
Project root: /path/to/project-root
Available models:
  - logistic_regression: ✓ Found
  - linear_svm: ✓ Found
  - random_forest: ✓ Found
  - mlp: ✓ Found
 * Running on http://0.0.0.0:5000
```

Keep this terminal running.

## Step 3: Setup React Frontend

### 3.1 Install Frontend Dependencies

Open a new terminal window:

```bash
cd frontend
npm install
```

### 3.2 Start the React Development Server

```bash
npm start
```

Expected output:
```
webpack compiled successfully
Local:            http://localhost:3000
On Your Network:  http://192.168.x.x:3000
```

The browser will automatically open to `http://localhost:3000`

## Step 4: Using the Application

### Demo Tab

1. **Load an Article**:
   - Paste your article in the text area
   - Or click one of the "Sample" buttons for quick testing
   - Minimum 100 characters required

2. **Generate Summaries**:
   - Click "✨ Generate Summaries"
   - Wait for processing (first load: 2-5 seconds, cached models: <2 seconds)

3. **View Results**:
   - 4 summary cards displayed side-by-side
   - Each shows model name, summary text, and word count
   - Click "📋 Copy" to copy summary to clipboard

### Comparison Tab

1. **Access Comparison Dashboard**:
   - Click "Comparison" tab
   - Automatically populates after generating summaries

2. **Analyze Metrics**:
   - **Word Count Chart**: Visual comparison of summary lengths
   - **Compression Ratio**: % of original article retained
   - **Radar Chart**: Multi-dimensional model performance
   - **Metrics Table**: Detailed statistics for each model

3. **Compare Summaries**:
   - All 4 summaries displayed side-by-side
   - Easy to spot differences in model outputs

## API Endpoints Reference

### Test the API

```bash
# Health check
curl http://localhost:5000/api/health

# List models
curl http://localhost:5000/api/models

# Generate summaries
curl -X POST http://localhost:5000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"article": "Your article text here..."}'
```

### Response Format

```json
{
  "article": "Original article text...",
  "summaries": {
    "logistic_regression": {
      "summary": "Generated summary text...",
      "sentences": [0, 2, 4]
    },
    "linear_svm": {...},
    "random_forest": {...},
    "mlp": {...}
  },
  "errors": {}
}
```

## Performance Characteristics

### Training Times (15k dataset, ~430k sentences)

| Model | Time | RAM | CPU |
|-------|------|-----|-----|
| Logistic Regression | 2-3 min | 2GB | 1-2 cores |
| Linear SVM | 2-3 min | 2GB | 1-2 cores |
| Random Forest | 60-90 min | 3GB | 2-4 cores |
| MLP | 120-180 min | 2.5GB | 2-4 cores |

### Inference Times (per article)

| Model | Time | First Load |
|-------|------|-----------|
| Logistic Regression | <100ms | 1-2 sec |
| Linear SVM | <100ms | 1-2 sec |
| Random Forest | 100-500ms | 5-10 sec |
| MLP | 50-200ms | 3-5 sec |

## Directory Structure

```
project-root/
├── scripts/
│   ├── train_extractive_model.py     # Main training script
│   ├── train_all_15k.sh              # Batch training script
│   ├── demo_run.py                   # CLI demo
│   └── ...
├── api/
│   ├── app.py                        # Flask backend
│   ├── requirements.txt              # Python dependencies
│   ├── README.md                     # API documentation
│   └── __init__.py
├── frontend/
│   ├── src/
│   │   ├── App.js                    # Main React component
│   │   ├── components/
│   │   │   ├── ArticleInput.js      # Input component
│   │   │   ├── SummaryDisplay.js    # Summary cards
│   │   │   └── ComparisonDashboard.js # Charts & metrics
│   │   └── ...
│   ├── public/
│   │   └── index.html
│   ├── package.json
│   └── README.md
├── experiments/
│   ├── logistic_regression_15k_model.pkl
│   ├── linear_svm_15k_model.pkl
│   ├── random_forest_15k_model.pkl
│   └── mlp_15k_model.pkl
└── ...
```

## Troubleshooting

### Issue: "Connection refused" when loading models

**Solution**: Ensure Flask API is running:
```bash
cd api
python app.py
```

### Issue: "X has N features, but Model is expecting M features"

**Solution**: Models were trained with different TF-IDF feature counts. Retrain with `--max-tfidf-features 8000` for consistency.

### Issue: Slow performance on first request

**Solution**: This is normal - models are being loaded into memory. Subsequent requests are faster due to caching.

### Issue: "CORS error" or "Blocked by CORS policy"

**Solution**: The Flask API should have CORS enabled. If not, restart `app.py`.

### Issue: Out of memory during training

**Solution**: 
- Reduce `--train-limit` to 5000-10000
- Close other applications
- Consider using a machine with more RAM

### Issue: Models not found on startup

**Solution**: Ensure models are in the correct directory:
```bash
ls -la experiments/*_15k_model.pkl
```

## Advanced Configuration

### Custom Model Parameters

Edit `scripts/train_extractive_model.py` to modify:
- `--max-tfidf-features`: TF-IDF vocabulary size
- `--n-estimators`: Number of trees for Random Forest
- `--hidden-layer-sizes`: MLP layer configuration

### Production Deployment

For production:

1. **Backend**: Use Gunicorn instead of Flask dev server
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Frontend**: Build for production
   ```bash
   cd frontend
   npm run build
   ```

3. **Nginx**: Configure reverse proxy for SSL

## Next Steps

- Try different articles to see how models compare
- Export summaries for analysis
- Fine-tune model parameters for your specific use case
- Deploy to cloud (AWS, GCP, Azure)

## Support & Documentation

- Frontend README: `frontend/README.md`
- API Documentation: `api/README.md`
- Original README: `README.md`

## License

See main project LICENSE file.
