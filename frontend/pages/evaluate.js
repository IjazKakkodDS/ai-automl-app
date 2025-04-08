// pages/evaluate.js

import React, { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import api from '../lib/api';
import styles from '../styles/Evaluate.module.css';

export default function Evaluate() {
  // Choose between using an uploaded CSV ("original") or stored preprocessed data ("preprocessed")
  const [datasetOption, setDatasetOption] = useState('original');
  const [selectedFile, setSelectedFile] = useState(null);
  const [modelList, setModelList] = useState([]);
  const [modelFile, setModelFile] = useState('');
  const [hasTarget, setHasTarget] = useState(false);
  const [targetColName, setTargetColName] = useState('target');
  const [evaluationResult, setEvaluationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Fetch session-specific model list; use session_id from localStorage if available
  useEffect(() => {
    async function fetchModelList() {
      try {
        const session_id = localStorage.getItem('session_id'); // Make sure to store session_id during model training
        const res = await api.get('/evaluation/list-models/', { params: { session_id } });
        if (res.data.status === 'success') {
          setModelList(res.data.models);
          if (res.data.models.length > 0) {
            setModelFile(res.data.models[0]);
          }
        }
      } catch (err) {
        console.error('Error fetching model list:', err);
      }
    }
    fetchModelList();
  }, []);

  // Handle file input change for "original" dataset option.
  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  // Evaluate the selected model on new data.
  const evaluateModel = async () => {
    setErrorMessage('');
    setEvaluationResult(null);

    if (datasetOption === 'preprocessed') {
      const datasetId = localStorage.getItem('dataset_id');
      if (!datasetId) {
        setErrorMessage('No preprocessed dataset found. Please run Preprocessing first.');
        return;
      }
      try {
        setLoading(true);
        const formData = new FormData();
        formData.append('model_file', modelFile);
        formData.append('has_target', hasTarget);
        formData.append('target_col', targetColName);
        const response = await api.post('/evaluation/evaluate/', formData, {
          params: { dataset_id: datasetId },
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        setEvaluationResult(response.data);
      } catch (err) {
        console.error(err);
        setErrorMessage(err.response?.data?.detail || 'Evaluation failed. Please try again.');
      } finally {
        setLoading(false);
      }
    } else {
      // For "original" dataset, require a file upload.
      if (!selectedFile) {
        setErrorMessage("Please upload a CSV file with your new data.");
        return;
      }
      try {
        setLoading(true);
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model_file', modelFile);
        formData.append('has_target', hasTarget);
        formData.append('target_col', targetColName);
        const response = await api.post('/evaluation/evaluate/', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        setEvaluationResult(response.data);
      } catch (err) {
        console.error(err);
        setErrorMessage(err.response?.data?.detail || 'Evaluation failed. Please try again.');
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <Layout prevLink="/forecasting" nextLink="/insights">
      <div className={styles.container}>
        <h1 className={styles.title}>Model Evaluation</h1>
        <p className={styles.description}>
          Evaluate your trained model on new data. Choose whether to use your preprocessed dataset or upload a new CSV file.
          Then, select the model you want to evaluate and indicate if your new data includes the target column.
        </p>
        <div className={styles.form}>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Dataset Source:</label>
            <select
              className={styles.select}
              value={datasetOption}
              onChange={(e) => setDatasetOption(e.target.value)}
            >
              <option value="original">Upload New CSV</option>
              <option value="preprocessed">Use Preprocessed Data</option>
            </select>
          </div>
          {datasetOption === 'original' && (
            <div className={styles.inputGroup}>
              <label className={styles.label}>CSV File:</label>
              <input
                type="file"
                className={styles.inputFile}
                onChange={handleFileChange}
              />
              <small className={styles.helpText}>
                Upload the CSV file containing your new data.
              </small>
            </div>
          )}
          <div className={styles.inputGroup}>
            <label className={styles.label}>Select Model:</label>
            <select
              className={styles.select}
              value={modelFile}
              onChange={(e) => setModelFile(e.target.value)}
            >
              {modelList.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <small className={styles.helpText}>
              Only models trained for this specific analysis are shown.
            </small>
          </div>
          <div className={styles.inputGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={hasTarget}
                onChange={(e) => setHasTarget(e.target.checked)}
                style={{ marginRight: '6px' }}
              />
              New data includes a target column?
            </label>
          </div>
          {hasTarget && (
            <div className={styles.inputGroup}>
              <label className={styles.label}>Target Column Name:</label>
              <input
                type="text"
                className={styles.input}
                value={targetColName}
                onChange={(e) => setTargetColName(e.target.value)}
                placeholder="e.g. Sales, Price, etc."
              />
              <small className={styles.helpText}>
                Ensure the target column name matches what was used during training.
              </small>
            </div>
          )}
          <button
            className={styles.button}
            onClick={evaluateModel}
            disabled={loading}
          >
            {loading ? 'Evaluating...' : 'Evaluate Model'}
          </button>
          {errorMessage && <div className={styles.error}>{errorMessage}</div>}
        </div>
        {evaluationResult && (
          <div className={styles.results}>
            <h2 className={styles.resultsTitle}>Evaluation Results</h2>
            <pre className={styles.resultsText}>
              {JSON.stringify(evaluationResult.results, null, 2)}
            </pre>
            {evaluationResult.results?.metrics && (
              <div className={styles.metrics}>
                <h3 className={styles.plotTitle}>Evaluation Metrics</h3>
                <pre className={styles.resultsText}>
                  {JSON.stringify(evaluationResult.results.metrics, null, 2)}
                </pre>
              </div>
            )}
            {evaluationResult.results?.column_info && (
              <div className={styles.metrics}>
                <h3 className={styles.plotTitle}>Column Information</h3>
                <pre className={styles.resultsText}>
                  {JSON.stringify(evaluationResult.results.column_info, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </Layout>
  );
}
