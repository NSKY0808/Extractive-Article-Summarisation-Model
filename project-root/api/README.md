# Extractive Summarization API

Flask REST API for the extractive article summarization system. Provides endpoints for model inference and summary generation.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure trained models exist in `../experiments/`:
   - `logistic_regression_15k_model.pkl`
   - `linear_svm_15k_model.pkl`
   - `random_forest_15k_model.pkl`
   - `mlp_15k_model.pkl`

3. Run the API:

```bash
python app.py
```

The API will start on `http://localhost:5000`

## API Documentation

### Health Check

```
GET /api/health
```

Response:
```json
{
  "status": "ok"
}
```

### List Models

```
GET /api/models
```

Response:
```json
{
  "logistic_regression": {
    "available": true,
    "path": "/path/to/model.pkl"
  },
  ...
}
```

### Generate Summaries

```
POST /api/summarize
Content-Type: application/json

{
  "article": "Your article text here...",
  "top_n_sentences": 3,
  "redundancy_threshold": 0.8,
  "mmr_lambda": 0.85,
  "max_candidates": 15
}
```

Response:
```json
{
  "article": "Your article text...",
  "summaries": {
    "logistic_regression": {
      "summary": "Generated summary...",
      "sentences": [0, 2, 4]
    },
    "linear_svm": {...},
    "random_forest": {...},
    "mlp": {...}
  },
  "errors": {}
}
```

### Comparison Endpoint

```
POST /api/compare
Content-Type: application/json

{
  "article": "Your article text here..."
}
```

Response:
```json
{
  "article": "Article preview (first 500 chars)...",
  "summaries": {
    "logistic_regression": "Summary...",
    "linear_svm": "Summary...",
    ...
  },
  "metrics": {
    "summary_lengths": {...},
    "article_length": 250
  }
}
```

## Configuration

All models are loaded from the `experiments/` directory. Model paths are defined in the `MODEL_PATHS` dictionary in `app.py`.

To use different models, update the paths:

```python
MODEL_PATHS = {
    "model_name": "path/to/model.pkl",
    ...
}
```

## Requirements

- flask>=2.3.0
- flask-cors>=4.0.0
- scikit-learn>=1.3.0
- scipy>=1.11.0
- datasets>=2.14.0
- huggingface-hub>=0.17.0
- numpy>=1.24.0

## Performance Notes

- Models are cached in memory after first load
- First request may take 2-5 seconds as models are loaded
- Subsequent requests are faster
- Large articles (>5000 words) may take longer to process
- Random Forest and MLP models are slower than Logistic Regression

## Error Handling

- 400: Invalid request format (missing `article` field)
- 400: Empty article text
- 500: Model loading or inference error

Errors are returned in the `errors` field of the response for each model.

## CORS

The API enables CORS for all origins. To restrict CORS, modify:

```python
CORS(app, origins=["http://localhost:3000"])
```

## Production Deployment

For production:

1. Change `debug=False` in `app.py`
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
3. Set up reverse proxy (Nginx) for SSL
4. Configure firewall and security headers

## License

Same as parent project
