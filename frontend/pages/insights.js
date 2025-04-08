// pages/insights.js

import React, { useState } from 'react';
import Layout from '../components/Layout';
import api from '../lib/api';
import styles from '../styles/Insights.module.css';

export default function Insights() {
  const [summaryOption, setSummaryOption] = useState('manual');
  const [edaSummary, setEdaSummary] = useState('Enter your EDA summary here...');
  const [modelSummary, setModelSummary] = useState('Enter your model training summary here...');
  const [modelChoice, setModelChoice] = useState('mistral');
  const [insights, setInsights] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleGenerateInsights = async () => {
    setLoading(true);
    setErrorMessage('');
    setInsights('');
    try {
      const formData = new FormData();
      if (summaryOption === 'preprocessed') {
        const datasetId = localStorage.getItem('dataset_id');
        if (!datasetId) {
          setErrorMessage("No datasetId found in localStorage. Please run Preprocessing first.");
          setLoading(false);
          return;
        }
        formData.append('eda_summary', '');
        formData.append('model_summary', '');
        formData.append('model_choice', modelChoice);
        const response = await api.post('/ai-insights/generate/', formData, {
          params: { dataset_id: datasetId },
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setInsights(response.data.insights);
      } else {
        formData.append('eda_summary', edaSummary || '');
        formData.append('model_summary', modelSummary || '');
        formData.append('model_choice', modelChoice);
        const response = await api.post('/ai-insights/generate/', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setInsights(response.data.insights);
      }
    } catch (err) {
      console.error(err);
      setErrorMessage("Error generating AI insights. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout prevLink="/evaluate" nextLink="/rag">
      <div className={styles.container}>
        <h1 className={styles.title}>AI Insights</h1>
        <p className={styles.description}>
          Generate AI-powered insights by combining your analysis. You can either enter your summaries manually or use preprocessed summaries.
        </p>
        <div className={styles.toggleContainer}>
          <label>Summary Source:</label>
          <select value={summaryOption} onChange={(e) => setSummaryOption(e.target.value)}>
            <option value="manual">Enter Manually</option>
            <option value="preprocessed">Preprocessed Summaries</option>
          </select>
        </div>
        {summaryOption === 'manual' && (
          <div className={styles.form}>
            <div className={styles.inputGroup}>
              <label className={styles.label}>EDA Summary:</label>
              <textarea
                rows={3}
                value={edaSummary}
                onChange={(e) => setEdaSummary(e.target.value)}
                className={styles.textarea}
                placeholder="Paste your EDA summary here"
              />
            </div>
            <div className={styles.inputGroup}>
              <label className={styles.label}>Model Summary:</label>
              <textarea
                rows={3}
                value={modelSummary}
                onChange={(e) => setModelSummary(e.target.value)}
                className={styles.textarea}
                placeholder="Paste your model training summary here"
              />
            </div>
          </div>
        )}
        <div className={styles.inputGroup}>
          <label className={styles.label}>Model Choice:</label>
          <select
            value={modelChoice}
            onChange={(e) => setModelChoice(e.target.value)}
            className={styles.select}
          >
            <option value="mistral">mistral (recommended for low memory)</option>
            <option value="gemma2">gemma2</option>
            <option value="llama3.3">llama3.3</option>
            <option value="llama2">llama2</option>
            <option value="gpt-4">gpt-4 (requires high memory)</option>
          </select>
        </div>
        <button onClick={handleGenerateInsights} disabled={loading} className={styles.button}>
          {loading ? "Generating Insights..." : "Generate AI Insights"}
        </button>
        {errorMessage && <div className={styles.error}>{errorMessage}</div>}
        {insights && (
          <div className={styles.results}>
            <h2 className={styles.resultsTitle}>AI Insights</h2>
            <pre className={styles.resultsText}>{insights}</pre>
          </div>
        )}
      </div>
    </Layout>
  );
}
