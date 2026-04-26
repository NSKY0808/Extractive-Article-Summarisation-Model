# Extractive Summarization Demo - React Frontend

A modern React-based frontend for the extractive article summarization system. Compare summaries from 4 different machine learning models (Logistic Regression, Linear SVM, Random Forest, MLP) with an interactive dashboard.

## Features

- 📝 **Article Input Interface**: Paste or type articles for summarization
- 📋 **Multi-Model Summaries**: Generate summaries from all 4 trained models simultaneously
- 📊 **Comparison Dashboard**: Visual comparison of model outputs with:
  - Word count analysis
  - Compression ratio metrics
  - Radar charts for performance overview
  - Detailed metrics table
- 💾 **Copy & Share**: Easily copy summaries to clipboard
- 🎨 **Modern UI**: Beautiful gradient design with responsive layout
- ⚡ **Real-time Processing**: Fast feedback from the backend API

## Setup Instructions

### Prerequisites

- Node.js (v14+)
- npm or yarn
- Python Flask API running (see Backend Setup below)

### Installation

1. Install frontend dependencies:

```bash
cd frontend
npm install
```

2. Ensure the Flask API is running on `http://localhost:5000`

3. Start the React development server:

```bash
npm start
```

The app will open at `http://localhost:3000`

## Backend API Setup

The frontend communicates with a Flask backend. To set up the backend:

1. Install Python dependencies:

```bash
cd api
pip install -r requirements.txt
```

2. Ensure the trained models are in `experiments/` directory:
   - `logistic_regression_15k_model.pkl`
   - `linear_svm_15k_model.pkl`
   - `random_forest_15k_model.pkl`
   - `mlp_15k_model.pkl`

3. Start the Flask API:

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/models` - List available models
- `POST /api/summarize` - Generate summaries for an article
  - Request body: `{ "article": "article text" }`
  - Response: Object with summaries from all models
- `POST /api/compare` - Generate summaries and comparison metrics
  - Request body: `{ "article": "article text" }`
  - Response: Detailed comparison data

## Project Structure

```
frontend/
├── public/
│   └── index.html          # HTML entry point
├── src/
│   ├── App.js              # Main app component
│   ├── App.css             # App styles
│   ├── index.js            # React entry point
│   ├── index.css           # Global styles
│   └── components/
│       ├── ArticleInput.js # Article input component
│       ├── ArticleInput.css
│       ├── SummaryDisplay.js # Summary cards component
│       ├── SummaryDisplay.css
│       ├── ComparisonDashboard.js # Charts & metrics
│       └── ComparisonDashboard.css
├── package.json
└── .gitignore
```

## Usage

1. **Demo Tab**: 
   - Enter or paste an article
   - Click "Generate Summaries"
   - View summaries from all 4 models
   - Copy any summary to clipboard

2. **Comparison Tab**:
   - View visual comparison of models
   - Charts show word counts and compression ratios
   - Radar chart displays performance metrics
   - Detailed metrics table with compression ratios
   - Side-by-side summary comparison

## Building for Production

```bash
npm run build
```

Creates an optimized production build in the `build/` directory.

## Technologies Used

- **React 18**: UI framework
- **Recharts**: Data visualization
- **Axios**: HTTP client for API calls
- **React Icons**: Icon library
- **CSS3**: Styling with gradients and animations

## Performance Tips

- The first API call may take longer as models are loaded
- Subsequent requests are faster due to model caching
- For large articles (>5000 words), generation may take 5-30 seconds depending on the model

## Troubleshooting

### "Connection refused" error
- Ensure Flask API is running on port 5000
- Check `proxy` setting in `package.json` is correct

### Models not found
- Verify trained models exist in `experiments/` directory
- Check file permissions and paths

### Slow performance
- CPU-intensive operations in the backend
- Consider using faster hardware or reducing dataset size for training

## License

Same as parent project
