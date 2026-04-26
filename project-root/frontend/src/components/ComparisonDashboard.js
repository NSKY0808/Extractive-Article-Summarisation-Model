import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import './ComparisonDashboard.css';

const CHART_COLORS = {
  primary: '#72685d',
  secondary: '#9b9287',
  tertiary: '#c2baaf',
  grid: '#e9e1d6',
  text: '#62584d',
  surface: '#fbf8f2',
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

function buildMetrics(article, summaries) {
  if (!article || !summaries?.summaries) {
    return null;
  }

  const articleWords = article.split(/\s+/).length;
  const articleSentences = (article.match(/[.!?]+/g) || []).length;

  const metricsData = {
    totalWords: articleWords,
    totalSentences: articleSentences,
    models: {},
  };

  Object.entries(summaries.summaries).forEach(([modelName, summary]) => {
    const summaryWords = summary.summary.split(/\s+/).length;
    const summarySentences = (summary.summary.match(/[.!?]+/g) || []).length;
    const compressionRatio = ((1 - summaryWords / articleWords) * 100).toFixed(1);

    metricsData.models[modelName] = {
      words: summaryWords,
      sentences: summarySentences,
      compressionRatio: parseFloat(compressionRatio),
      summary: summary.summary,
    };
  });

  return metricsData;
}

function buildInsights(metrics, benchmarkMetrics) {
  const modelEntries = Object.entries(metrics.models);
  const benchmarkEntries = Object.entries(benchmarkMetrics?.models || {});
  const shortest = modelEntries.reduce((best, current) =>
    !best || current[1].words < best[1].words ? current : best
  , null);
  const densest = modelEntries.reduce((best, current) =>
    !best || current[1].compressionRatio > best[1].compressionRatio ? current : best
  , null);
  const strongest = benchmarkEntries.reduce((best, current) =>
    !best || (current[1].test?.rouge_1 || 0) > (best[1].test?.rouge_1 || 0) ? current : best
  , null);

  return [
    shortest && {
      label: 'Shortest summary',
      value: MODEL_LABELS[shortest[0]],
      detail: `${shortest[1].words} words`,
    },
    densest && {
      label: 'Most compressed',
      value: MODEL_LABELS[densest[0]],
      detail: `${densest[1].compressionRatio}% compression`,
    },
    strongest && {
      label: 'Best benchmark model',
      value: MODEL_LABELS[strongest[0]],
      detail: `ROUGE-1 ${formatScore(strongest[1].test?.rouge_1)}`,
    },
  ].filter(Boolean);
}

function ComparisonDashboard({ article, summaries, benchmarkMetrics }) {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    setMetrics(buildMetrics(article, summaries));
  }, [article, summaries]);

  if (!metrics) {
    return (
      <div className="comparison-placeholder">
        <p>Enter an article and generate summaries to see the comparison dashboard.</p>
      </div>
    );
  }

  const wordCountData = Object.entries(metrics.models).map(([model, data]) => ({
    model: MODEL_LABELS[model],
    words: data.words,
  }));

  const compressionData = Object.entries(metrics.models).map(([model, data]) => ({
    model: MODEL_LABELS[model],
    compression: data.compressionRatio,
  }));

  const radarData = Object.entries(metrics.models).map(([model, data]) => ({
    model: MODEL_LABELS[model],
    words: (data.words / metrics.totalWords) * 100,
    sentences: (data.sentences / Math.max(metrics.totalSentences, 1)) * 100,
    compression: data.compressionRatio,
  }));

  const benchmarkRows = Object.entries(benchmarkMetrics?.models || {}).map(([modelName, modelData]) => ({
    modelName,
    validationF1: modelData.validation?.f1,
    validationRouge1: modelData.validation?.rouge_1,
    testRouge1: modelData.test?.rouge_1,
    testRouge2: modelData.test?.rouge_2,
    testRougeL: modelData.test?.rouge_l,
  }));

  const benchmarkChartData = benchmarkRows.map((row) => ({
    model: MODEL_LABELS[row.modelName],
    'Test ROUGE-1': row.testRouge1,
    'Test ROUGE-2': row.testRouge2,
    'Test ROUGE-L': row.testRougeL,
  }));

  const insights = buildInsights(metrics, benchmarkMetrics);
  const articlePreview = article.split(/\s+/).slice(0, 70).join(' ');

  return (
    <div className="comparison-dashboard">
      <div className="dashboard-hero">
        <div className="dashboard-header">
          <h2>Summary Comparison Dashboard</h2>
          <p className="dashboard-subtitle">
            A cleaner comparison view for live output, compression behavior, and tracked benchmark performance.
          </p>
          <div className="article-stats">
            <div className="stat">
              <span className="stat-label">Original Words</span>
              <span className="stat-value">{metrics.totalWords}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Original Sentences</span>
              <span className="stat-value">{metrics.totalSentences}</span>
            </div>
            {benchmarkMetrics?.run_date && (
              <div className="stat">
                <span className="stat-label">Benchmark Run</span>
                <span className="stat-value">{benchmarkMetrics.run_date}</span>
              </div>
            )}
          </div>
        </div>

        <aside className="comparison-aside">
          <div className="aside-card">
            <div className="section-overline">Article preview</div>
            <p>{articlePreview}{article.split(/\s+/).length > 70 ? '...' : ''}</p>
          </div>
          <div className="aside-card">
            <div className="section-overline">Quick insights</div>
            <div className="insight-stack">
              {insights.map((insight) => (
                <div key={insight.label} className="insight-row">
                  <span>{insight.label}</span>
                  <strong>{insight.value}</strong>
                  <small>{insight.detail}</small>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>

      <div className="charts-container">
        <div className="chart-wrapper chart-large">
          <h3>Summary Word Count</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={wordCountData} margin={{ top: 20, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid stroke={CHART_COLORS.grid} vertical={false} />
              <XAxis dataKey="model" tick={{ fill: CHART_COLORS.text, fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: CHART_COLORS.text, fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: CHART_COLORS.surface,
                  border: '1px solid #e6ddd1',
                  borderRadius: '16px',
                  color: '#312a23',
                }}
              />
              <Bar dataKey="words" fill={CHART_COLORS.primary} radius={[10, 10, 4, 4]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-wrapper">
          <h3>Compression Ratio</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={compressionData} margin={{ top: 20, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid stroke={CHART_COLORS.grid} vertical={false} />
              <XAxis dataKey="model" tick={{ fill: CHART_COLORS.text, fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: CHART_COLORS.text, fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: CHART_COLORS.surface,
                  border: '1px solid #e6ddd1',
                  borderRadius: '16px',
                  color: '#312a23',
                }}
              />
              <Bar dataKey="compression" fill={CHART_COLORS.secondary} radius={[10, 10, 4, 4]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-wrapper">
          <h3>Live Summary Retention</h3>
          <ResponsiveContainer width="100%" height={360}>
            <RadarChart data={radarData}>
              <PolarGrid stroke={CHART_COLORS.grid} />
              <PolarAngleAxis dataKey="model" tick={{ fill: CHART_COLORS.text, fontSize: 11 }} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: CHART_COLORS.text, fontSize: 11 }} />
              <Radar
                name="Retention %"
                dataKey="words"
                stroke={CHART_COLORS.primary}
                fill={CHART_COLORS.primary}
                fillOpacity={0.24}
              />
              <Radar
                name="Sentence Retention %"
                dataKey="sentences"
                stroke={CHART_COLORS.secondary}
                fill={CHART_COLORS.secondary}
                fillOpacity={0.18}
              />
              <Radar
                name="Compression %"
                dataKey="compression"
                stroke={CHART_COLORS.tertiary}
                fill={CHART_COLORS.tertiary}
                fillOpacity={0.22}
              />
              <Legend wrapperStyle={{ color: CHART_COLORS.text, fontSize: 12 }} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {benchmarkChartData.length > 0 && (
          <div className="chart-wrapper">
            <h3>Tracked Benchmark ROUGE</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={benchmarkChartData} margin={{ top: 20, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid stroke={CHART_COLORS.grid} vertical={false} />
                <XAxis dataKey="model" tick={{ fill: CHART_COLORS.text, fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 0.35]} tick={{ fill: CHART_COLORS.text, fontSize: 12 }} axisLine={false} tickLine={false} />
                <Tooltip
                  formatter={(value) => formatScore(value)}
                  contentStyle={{
                    backgroundColor: CHART_COLORS.surface,
                    border: '1px solid #e6ddd1',
                    borderRadius: '16px',
                    color: '#312a23',
                  }}
                />
                <Legend wrapperStyle={{ color: CHART_COLORS.text, fontSize: 12 }} />
                <Bar dataKey="Test ROUGE-1" fill={CHART_COLORS.primary} radius={[8, 8, 4, 4]} />
                <Bar dataKey="Test ROUGE-2" fill={CHART_COLORS.secondary} radius={[8, 8, 4, 4]} />
                <Bar dataKey="Test ROUGE-L" fill={CHART_COLORS.tertiary} radius={[8, 8, 4, 4]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <div className="metrics-table">
        <h3>Live Summary Metrics</h3>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Words</th>
              <th>Sentences</th>
              <th>Compression %</th>
              <th>Compression Ratio</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(metrics.models).map(([model, data]) => (
              <tr key={model}>
                <td className="model-name">{MODEL_LABELS[model]}</td>
                <td>{data.words}</td>
                <td>{data.sentences}</td>
                <td>
                  <span className="compression-badge">{data.compressionRatio}%</span>
                </td>
                <td>1:{(metrics.totalWords / data.words).toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {benchmarkRows.length > 0 && (
        <div className="metrics-table">
          <h3>Tracked Benchmark Metrics</h3>
          <p className="metrics-note">
            These scores match the latest benchmark summary used by the README and provide a stable reference next to the live article comparison.
          </p>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Val F1</th>
                <th>Val ROUGE-1</th>
                <th>Test ROUGE-1</th>
                <th>Test ROUGE-2</th>
                <th>Test ROUGE-L</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkRows.map((row) => (
                <tr key={row.modelName}>
                  <td className="model-name">{MODEL_LABELS[row.modelName]}</td>
                  <td>{formatScore(row.validationF1)}</td>
                  <td>{formatScore(row.validationRouge1)}</td>
                  <td>{formatScore(row.testRouge1)}</td>
                  <td>{formatScore(row.testRouge2)}</td>
                  <td>{formatScore(row.testRougeL)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="summaries-comparison">
        <h3>All Summaries for Comparison</h3>
        <div className="summaries-list">
          {Object.entries(metrics.models).map(([model, data]) => (
            <div key={model} className="summary-item">
              <h4>{MODEL_LABELS[model]}</h4>
              <p>{data.summary}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default ComparisonDashboard;
