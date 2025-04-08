// pages/forecasting.js

import React, { useState } from 'react';
import Layout from '../components/Layout';
import api from '../lib/api';
import styles from '../styles/Forecasting.module.css';

export default function Forecasting() {
  const [datasetOption, setDatasetOption] = useState('original');
  const [selectedFile, setSelectedFile] = useState(null);
  const [forecastModel, setForecastModel] = useState('prophet'); // 'prophet' or 'arima'
  const [targetCol, setTargetCol] = useState('Sales');
  const [forecastPeriod, setForecastPeriod] = useState(30);
  const [forecastResult, setForecastResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const runForecast = async () => {
    setErrorMessage('');
    setForecastResult(null);
    setLoading(true);
    try {
      const params = {
        forecast_model: forecastModel,
        target_col: targetCol,
        forecast_period: forecastPeriod,
      };
      if (datasetOption === 'preprocessed') {
        const datasetId = localStorage.getItem('dataset_id');
        if (!datasetId) {
          setErrorMessage('No dataset_id found in localStorage. Please run Preprocessing first.');
          setLoading(false);
          return;
        }
        params.dataset_id = datasetId;
        params.folder = 'processed_data';
        const response = await api.post('/forecasting/forecast/', null, { params });
        setForecastResult(response.data);
      } else if (datasetOption === 'original') {
        // For original dataset, try to retrieve the stored raw dataset ID.
        const rawDatasetId = localStorage.getItem('raw_dataset_id');
        if (rawDatasetId) {
          params.dataset_id = rawDatasetId;
          params.folder = 'original_data';
          const response = await api.post('/forecasting/forecast/', null, { params });
          setForecastResult(response.data);
        } else {
          if (!selectedFile) {
            setErrorMessage('Please select a CSV file for the original dataset.');
            setLoading(false);
            return;
          }
          const formData = new FormData();
          formData.append('file', selectedFile);
          // For original data, even if file is uploaded, set folder to "original_data" in params.
          params.folder = 'original_data';
          const response = await api.post('/forecasting/forecast/', formData, {
            params,
            headers: { 'Content-Type': 'multipart/form-data' },
          });
          setForecastResult(response.data);
        }
      }
    } catch (err) {
      console.error(err);
      setErrorMessage(err.response?.data?.detail || err.message || 'Forecasting failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout prevLink="/model-training" nextLink="/evaluate">
      <div className={styles.container}>
        <h1 className={styles.title}>Time-Series Forecasting</h1>
        <p className={styles.description}>
          Upload a CSV file or use your preprocessed dataset. Choose a forecast model (Prophet or ARIMA),
          set the target column and horizon, then view predictions below.
        </p>
        <div className={styles.form}>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Dataset Source:</label>
            <select
              className={styles.select}
              value={datasetOption}
              onChange={(e) => setDatasetOption(e.target.value)}
            >
              <option value="original">Original Dataset</option>
              <option value="preprocessed">Preprocessed Dataset</option>
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
            </div>
          )}
        </div>
        <div className={styles.form}>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Forecast Model:</label>
            <select
              className={styles.select}
              value={forecastModel}
              onChange={(e) => setForecastModel(e.target.value)}
            >
              <option value="prophet">Prophet</option>
              <option value="arima">ARIMA</option>
            </select>
          </div>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Target Column:</label>
            <input
              type="text"
              className={styles.input}
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              placeholder="Enter target column name"
            />
          </div>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Forecast Period (Days):</label>
            <input
              type="number"
              className={styles.input}
              min="1"
              value={forecastPeriod}
              onChange={(e) => setForecastPeriod(parseInt(e.target.value, 10))}
            />
          </div>
          <button
            className={styles.button}
            onClick={runForecast}
            disabled={loading}
          >
            {loading ? 'Running Forecast...' : 'Run Forecast'}
          </button>
          {errorMessage && <div className={styles.error}>{errorMessage}</div>}
        </div>
        {forecastResult && (
          <div className={styles.results}>
            <h2 className={styles.resultsTitle}>Forecast Results</h2>
            <div className={styles.forecastSummary}>
              <p>
                <strong>Forecast Model:</strong> {forecastModel.toUpperCase()}
              </p>
              <p>
                <strong>Horizon:</strong> {forecastPeriod} days
              </p>
              <p>
                <strong>Target:</strong> {targetCol}
              </p>
            </div>
            <div className={styles.resultsContent}>
              {forecastResult.results && Array.isArray(forecastResult.results) && (
                <div className={styles.tableContainer}>
                  <table className={styles.resultsTable}>
                    <thead>
                      <tr>
                        <th>ds</th>
                        <th>yhat</th>
                        <th>yhat_lower</th>
                        <th>yhat_upper</th>
                      </tr>
                    </thead>
                    <tbody>
                      {forecastResult.results.map((row, idx) => (
                        <tr key={idx}>
                          <td>{row.ds}</td>
                          <td>{typeof row.yhat === 'number' ? row.yhat.toFixed(4) : row.yhat}</td>
                          <td>{typeof row.yhat_lower === 'number' ? row.yhat_lower.toFixed(4) : row.yhat_lower}</td>
                          <td>{typeof row.yhat_upper === 'number' ? row.yhat_upper.toFixed(4) : row.yhat_upper}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {forecastResult.forecast_plot && (
                <div className={styles.plotContainer}>
                  <h3 className={styles.plotTitle}>Forecast Plot</h3>
                  <img
                    src={`data:image/png;base64,${forecastResult.forecast_plot}`}
                    alt="Forecast Plot"
                    className={styles.plotImage}
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
