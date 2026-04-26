import React, { useEffect, useState } from 'react';
import './App.css';
import ArticleInput from './components/ArticleInput';
import SummaryDisplay from './components/SummaryDisplay';
import ComparisonDashboard from './components/ComparisonDashboard';

function App() {
  const [activeTab, setActiveTab] = useState('demo');
  const [article, setArticle] = useState('');
  const [summaries, setSummaries] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [benchmarkMetrics, setBenchmarkMetrics] = useState(null);

  useEffect(() => {
    const loadBenchmarkMetrics = async () => {
      try {
        const response = await fetch('/api/benchmark-metrics');
        if (!response.ok) {
          return;
        }

        const data = await response.json();
        setBenchmarkMetrics(data);
      } catch (err) {
        console.error('Failed to load benchmark metrics', err);
      }
    };

    loadBenchmarkMetrics();
  }, []);

  const handleSummarize = async (text) => {
    setArticle(text);
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ article: text }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setSummaries(data);
    } catch (err) {
      setError(err.message || 'Failed to generate summaries');
      setSummaries(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="ambient-orb ambient-orb-left" />
      <div className="ambient-orb ambient-orb-right" />

      <header className="app-header">
        <div className="eyebrow">Extractive summarization workspace</div>
        <h1>Compare four classical models in a calmer, more editorial interface.</h1>
        <p>
          Paste a news article, generate side-by-side summaries, and review live outputs against
          tracked benchmark metrics in one place.
        </p>
      </header>

      <div className="tab-navigation">
        <button
          className={`tab-btn ${activeTab === 'demo' ? 'active' : ''}`}
          onClick={() => setActiveTab('demo')}
        >
          Demo
        </button>
        <button
          className={`tab-btn ${activeTab === 'comparison' ? 'active' : ''}`}
          onClick={() => setActiveTab('comparison')}
        >
          Comparison
        </button>
      </div>

      <main className="app-main">
        {activeTab === 'demo' && (
          <div className="demo-section">
            <ArticleInput onSummarize={handleSummarize} loading={loading} />

            {error && (
              <div className="error-message">
                <strong>Error:</strong> {error}
              </div>
            )}

            {loading && (
              <div className="loading-spinner">
                <div className="spinner" />
                <p>Generating summaries...</p>
              </div>
            )}

            {summaries && !loading && (
              <SummaryDisplay summaries={summaries} benchmarkMetrics={benchmarkMetrics} />
            )}
          </div>
        )}

        {activeTab === 'comparison' && (
          <div className="comparison-section">
            <ComparisonDashboard
              article={article}
              summaries={summaries}
              benchmarkMetrics={benchmarkMetrics}
            />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>CNN/DailyMail • Logistic Regression • Linear SVM • Random Forest • MLP</p>
      </footer>
    </div>
  );
}

export default App;
