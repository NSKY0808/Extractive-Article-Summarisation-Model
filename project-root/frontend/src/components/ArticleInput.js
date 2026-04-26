import React, { useState } from 'react';
import './ArticleInput.css';

function ArticleInput({ onSummarize, loading }) {
  const [text, setText] = useState('');
  const [characterCount, setCharacterCount] = useState(0);

  const handleTextChange = (e) => {
    const value = e.target.value;
    setText(value);
    setCharacterCount(value.length);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      onSummarize(text);
    }
  };

  const handleClear = () => {
    setText('');
    setCharacterCount(0);
  };

  const sampleArticles = [
    `(CNN) Artificial intelligence has become one of the most transformative technologies of our time. From healthcare to finance, AI is changing how we work and live. However, this rapid advancement also raises important questions about ethics and privacy. Researchers are working hard to ensure that AI systems are fair and transparent. The future of AI depends on responsible development and deployment. Companies and governments must collaborate to create guidelines that protect individuals while fostering innovation. As we move forward, education and awareness will be key to helping society adapt to these changes.`,
    `Breaking News: Scientists discover new species in deep ocean. A team of marine biologists has identified three previously unknown species living in the Mariana Trench. The discovery was made during a research expedition funded by the National Geographic Society. These organisms have adapted to extreme pressure and darkness. The findings provide valuable insights into how life can exist in the most challenging environments. Researchers believe there may be many more undiscovered species in the deep ocean. This discovery highlights the importance of continued ocean exploration and conservation efforts.`,
  ];

  return (
    <div className="article-input-container">
      <div className="input-card">
        <div className="input-card-header">
          <div>
            <div className="section-label">Input</div>
            <h2>Bring in an article to summarize.</h2>
          </div>
          <p>
            Use your own article or start from a sample. Summaries are generated side by side across
            all four models.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="textarea-wrapper">
            <textarea
              value={text}
              onChange={handleTextChange}
              placeholder="Paste a news article here. A few paragraphs works best."
              rows={12}
              disabled={loading}
              className="article-textarea"
            />
            <div className="input-meta-row">
              <div className="input-hint">Minimum 100 characters</div>
              <div className="char-count">{characterCount} characters</div>
            </div>
          </div>

          <div className="button-group">
            <button
              type="submit"
              disabled={loading || characterCount < 100}
              className="btn btn-primary"
            >
              {loading ? 'Generating summaries...' : 'Generate summaries'}
            </button>
            <button
              type="button"
              onClick={handleClear}
              disabled={loading || !text}
              className="btn btn-secondary"
            >
              Clear
            </button>
          </div>
        </form>

        <div className="samples-section">
          <div className="section-label">Samples</div>
          <h3>Start with a prepared article.</h3>
          <div className="sample-buttons">
            {sampleArticles.map((sample, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setText(sample);
                  setCharacterCount(sample.length);
                }}
                disabled={loading}
                className="btn btn-sample"
              >
                Sample {idx + 1}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ArticleInput;
