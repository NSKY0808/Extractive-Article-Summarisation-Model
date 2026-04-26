# Quick Reference - Getting Started

## 🚀 Start Here

The 15k production models are ready to use! Just run one command.

### Option A: One-Command Startup (Recommended)

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
./start.sh
```

This will:
1. Check prerequisites
2. Install dependencies
3. Start Flask API (port 5000) with 15k models loaded
4. Start React frontend (port 3000)
5. Open browser to http://localhost:3000

### Option B: Step-by-Step Start

**Step 1 - Start Flask API (Terminal 1):**
```bash
cd project-root/api
pip install -r requirements.txt
python app.py
```

API will output which 15k models are loaded.

**Step 2 - Start React (Terminal 2):**
```bash
cd project-root/frontend
npm install
npm start
```

Access the app at http://localhost:3000

## 📊 15k Models Status

✅ **All four models trained on 15,000 records:**
- Logistic Regression (ROUGE-1: 0.288)
- Linear SVM (ROUGE-1: 0.289)
- Random Forest (ROUGE-1: 0.307) - Best performer
- MLP (ROUGE-1: 0.296)

Models located in: `project-root/experiments/*_15k_model.pkl`

## ♻️ Retraining Models (Optional)

To retrain all 15k models on Windows:

```bash
cd project-root
scripts\train_all_15k.bat
```

Expected duration: **4-8 hours total**
- Logistic Regression: 2-3 min
- Linear SVM: 2-3 min
- Random Forest: 60-90 min
- MLP: 120-180 min

To retrain on Linux/Mac:

```bash
cd project-root
chmod +x scripts/train_all_15k.sh
./scripts/train_all_15k.sh
```

## 🎯 Using the Application

### Demo Tab
1. Paste an article (min 100 characters)
2. Click "Generate Summaries"
3. View 4 summary cards side-by-side
4. Click "Copy" to save summary

### Comparison Tab
1. Generated summaries populate automatically
2. View multiple charts:
   - Word count comparison
   - Compression ratios
   - Multi-dimensional radar chart
3. Detailed metrics table
4. Side-by-side summary review

## 📁 Key Files

| File | Purpose |
|------|---------|
| `start.bat` / `start.sh` | One-command startup |
| `QUICK_START.md` | This file |
| `README.md` | Full documentation |
| `SETUP_GUIDE.md` | Detailed setup guide |
| `project-root/api/app.py` | Flask API with 15k models |
| `project-root/frontend/src/App.js` | React main app |
| `project-root/scripts/train_extractive_model.py` | Model training script |
| `project-root/experiments/*_15k_model.pkl` | Production models |


## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Port 3000 in use | `npm start` with different port or kill process |
| Port 5000 in use | `python app.py` with different port |
| Models not found | Run training scripts first |
| "Connection refused" | Ensure Flask API is running |
| CORS errors | Restart Flask API |

## 📚 Documentation

- **Full Setup**: Read `SETUP_GUIDE.md`
- **API Details**: See `api/README.md`
- **Frontend Details**: See `frontend/README.md`
- **What Was Built**: See `IMPLEMENTATION_SUMMARY.md`

## 🌐 URLs

- Frontend: http://localhost:3000
- API: http://localhost:5000
- API Health: http://localhost:5000/api/health

## ⏱️ Typical Workflow

1. **First Time Setup (1 hour):**
   ```bash
   # Train models (first time only)
   cd project-root
   python scripts/train_extractive_model.py --model-type logistic_regression --train-limit 15000 --validation-limit 2000 --output-model-path experiments/logistic_regression_15k_model.pkl --max-tfidf-features 8000
   # ... repeat for other 3 models
   ```

2. **Daily Usage (5 seconds):**
   ```bash
   start.bat  # or start.sh on Unix/Mac
   # App opens at http://localhost:3000
   ```

3. **Stop Services:**
   - Close the terminal windows or Ctrl+C

## 💡 Tips

- **Sample Articles**: Click "Sample 1" or "Sample 2" for quick testing
- **Copy Summaries**: Click "📋 Copy" button on any summary card
- **Analyze Differences**: Use the Comparison tab to see which model is best
- **First Load Slow**: Models cache after first load, subsequent requests are fast

## 📈 Model Performance (15k dataset)

| Model | ROUGE-1 | Training Time |
|-------|---------|---------------|
| Logistic Regression | ~0.25 | 2-3 min |
| Linear SVM | ~0.25 | 2-3 min |
| Random Forest | ~0.27 | 60-90 min |
| MLP | ~0.27 | 120-180 min |

## ✅ What's Included

- ✅ Full React frontend with modern UI
- ✅ Flask REST API backend
- ✅ 4 trained models (prepared for 15k training)
- ✅ Comparison dashboard with charts
- ✅ Startup scripts for Windows/Linux/Mac
- ✅ Complete documentation
- ✅ CLI demo mode
- ✅ Error handling & validation

## 🎓 Next Steps

1. Start the application (`start.bat` or `start.sh`)
2. Try the demo with sample articles
3. Use the comparison dashboard to evaluate models
4. Train on full 15k dataset (optional, time-consuming)
5. Deploy to production (see `SETUP_GUIDE.md`)

---

**Need help?** See `SETUP_GUIDE.md` for detailed instructions.
