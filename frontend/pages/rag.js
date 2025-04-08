// pages/rag.js

import React, { useState } from 'react';
import Layout from '../components/Layout';
import api from '../lib/api';
import styles from '../styles/Rag.module.css';

export default function Rag() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(3);
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleQuery = async () => {
    if (!query.trim()) {
      setErrorMessage("Please enter a query.");
      return;
    }
    setLoading(true);
    setErrorMessage('');
    setAnswer(null);
    try {
      const formData = new FormData();
      formData.append('query', query);
      formData.append('top_k', topK);
      const response = await api.post('/rag/query-summarize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setAnswer(response.data.answer);
    } catch (err) {
      console.error(err);
      setErrorMessage("Error processing query. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout prevLink="/insights" nextLink="/reports">
      <div className={styles.container}>
        <h1 className={styles.title}>RAG & Agentic AI</h1>
        <p className={styles.description}>
          Ask a question to retrieve relevant document snippets and get a concise, bullet-point summary.
        </p>
        <div className={styles.form}>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Your Question:</label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className={styles.input}
              placeholder="e.g., Why are negative R2 values observed?"
            />
          </div>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Top K Results:</label>
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value, 10))}
              className={styles.input}
              min="1"
            />
          </div>
          <button onClick={handleQuery} disabled={loading} className={styles.button}>
            {loading ? "Searching..." : "Search & Summarize"}
          </button>
          {errorMessage && <div className={styles.error}>{errorMessage}</div>}
        </div>
        {answer && (
          <div className={styles.results}>
            <h2 className={styles.resultsTitle}>Summarized Answer</h2>
            <pre className={styles.resultsText}>{answer}</pre>
          </div>
        )}
      </div>
    </Layout>
  );
}
