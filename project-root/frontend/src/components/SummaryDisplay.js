import React, { useState } from 'react';
import './SummaryDisplay.css';

const MODEL_TONES = {
  logistic_regression: {
    border: '#d6cec1',
    surface: '#fbf8f3',
    accent: '#81786d',
  },
  linear_svm: {
    border: '#cdc6bb',
    surface: '#f8f5ef',
    accent: '#72695f',
  },
  random_forest: {
    border: '#d8d0c5',
    surface: '#fcfaf6',
    accent: '#8a8073',
  },
  mlp: {
    border: '#d2cbc1',
    surface: '#f7f4ee',
    accent: '#6e665d',
  },
};

const MODEL_LABELS = {
  logistic_regression: 'Logistic Regression',
  linear_svm: 'Linear SVM',
  random_forest: 'Random Forest',
  mlp: 'MLP',
};

function formatScore(value) {
  return typeof value === 'number' ? value.toFixed(3) : 'N/A';
}

function wordCount(text) {
  return text.split(/\s+/).filter(Boolean).length;
}

function sentenceCount(text) {
  return (text.match(/[.!?]+/g) || []).length;
}

function SummaryDisplay({ summaries, benchmarkMetrics }) {
  const [activeSummary, setActiveSummary] = useState(null);

  if (!summaries || !summaries.summaries) {
    return null;
  }

  const availableSummaries = Object.keys(summaries.summaries);
  const benchmarkEntries = Object.entries(benchmarkMetrics?.models || {});
  const bestBenchmarkModel =
    benchmarkEntries.sort(
      (left, right) => (right[1].test?.rouge_1 || 0) - (left[1].test?.rouge_1 || 0)
    )[0]?.[0] || null;

  return (
    <div className="summary-display">
      <div className="summary-topbar">
        <div>
          <h2>Generated Summaries</h2>
          <p className="summary-subtitle">
            Neutral model cards with benchmark context and a dedicated reading view for each summary.
          </p>
        </div>
        {bestBenchmarkModel && (
          <div className="summary-insight">
            <span className="summary-insight-label">Best tracked benchmark</span>
            <strong>{MODEL_LABELS[bestBenchmarkModel]}</strong>
          </div>
        )}
      </div>

      {summaries.errors && Object.keys(summaries.errors).length > 0 && (
        <div className="errors-warning">
          Some models encountered errors:
          {Object.entries(summaries.errors).map(([model, error]) => (
            <div key={model} className="error-item">
              <strong>{model}:</strong> {error}
            </div>
          ))}
        </div>
      )}

      <div className="summaries-grid">
        {availableSummaries.map((modelName) => {
          const summary = summaries.summaries[modelName];
          const benchmark = benchmarkMetrics?.models?.[modelName];
          const tone = MODEL_TONES[modelName];
          const countWords = wordCount(summary.summary);
          const countSentences = sentenceCount(summary.summary);
          const isBestModel = bestBenchmarkModel === modelName;

          return (
            <article
              key={modelName}
              className="summary-card"
              style={{
                borderColor: tone.border,
                backgroundColor: tone.surface,
              }}
            >
              <div className="summary-card-accent" style={{ backgroundColor: tone.accent }} />

              <div className="summary-header">
                <div className="summary-heading-group">
                  <span className="summary-icon" style={{ color: tone.accent }}>
                    {MODEL_LABELS[modelName].slice(0, 2).toUpperCase()}
                  </span>
                  <div>
                    <h3>{MODEL_LABELS[modelName]}</h3>
                    <p className="summary-model-note">
                      {isBestModel ? 'Highest tracked test ROUGE-1 in the current benchmark.' : 'Classical extractive baseline.'}
                    </p>
                  </div>
                </div>
                {isBestModel && <span className="best-badge">Best benchmark</span>}
              </div>

              <div className="summary-content">
                <p>{summary.summary}</p>

                {benchmark && (
                  <div className="benchmark-strip">
                    <div className="benchmark-pill">
                      <span className="pill-label">Val F1</span>
                      <span className="pill-value">{formatScore(benchmark.validation?.f1)}</span>
                    </div>
                    <div className="benchmark-pill">
                      <span className="pill-label">Test R-1</span>
                      <span className="pill-value">{formatScore(benchmark.test?.rouge_1)}</span>
                    </div>
                    <div className="benchmark-pill">
                      <span className="pill-label">Test R-L</span>
                      <span className="pill-value">{formatScore(benchmark.test?.rouge_l)}</span>
                    </div>
                  </div>
                )}

                <div className="summary-meta">
                  <span className="word-count">{countWords} words</span>
                  <span className="sentence-count">{countSentences} sentences</span>
                </div>
              </div>

              <div className="summary-actions">
                <button
                  className="summary-btn"
                  onClick={() => setActiveSummary({ modelName, summary: summary.summary, benchmark })}
                  title="Open summary"
                >
                  Open
                </button>
                <button
                  className="summary-btn"
                  onClick={() => navigator.clipboard.writeText(summary.summary)}
                  title="Copy summary"
                >
                  Copy
                </button>
              </div>
            </article>
          );
        })}
      </div>

      {availableSummaries.length === 0 && (
        <div className="no-summaries">
          <p>No summaries available. Please try again.</p>
        </div>
      )}

      {benchmarkMetrics?.run_date && (
        <p className="benchmark-note">
          Benchmark metrics come from the tracked evaluation run dated {benchmarkMetrics.run_date}.
        </p>
      )}

      {activeSummary && (
        <div className="summary-modal-backdrop" onClick={() => setActiveSummary(null)} role="presentation">
          <div className="summary-modal" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true">
            <div className="summary-modal-header">
              <div>
                <div className="section-overline">Reading view</div>
                <h3>{MODEL_LABELS[activeSummary.modelName]}</h3>
              </div>
              <button className="summary-modal-close" onClick={() => setActiveSummary(null)}>
                Close
              </button>
            </div>

            {activeSummary.benchmark && (
              <div className="summary-modal-metrics">
                <div className="benchmark-pill">
                  <span className="pill-label">Val F1</span>
                  <span className="pill-value">{formatScore(activeSummary.benchmark.validation?.f1)}</span>
                </div>
                <div className="benchmark-pill">
                  <span className="pill-label">Test R-1</span>
                  <span className="pill-value">{formatScore(activeSummary.benchmark.test?.rouge_1)}</span>
                </div>
                <div className="benchmark-pill">
                  <span className="pill-label">Test R-L</span>
                  <span className="pill-value">{formatScore(activeSummary.benchmark.test?.rouge_l)}</span>
                </div>
              </div>
            )}

            <div className="summary-modal-body">
              <p>{activeSummary.summary}</p>
            </div>

            <div className="summary-modal-footer">
              <span>
                {wordCount(activeSummary.summary)} words • {sentenceCount(activeSummary.summary)} sentences
              </span>
              <button
                className="summary-btn summary-btn-dark"
                onClick={() => navigator.clipboard.writeText(activeSummary.summary)}
              >
                Copy summary
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default SummaryDisplay;
