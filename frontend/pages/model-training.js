// pages/model-training.js

import React, { useState, useEffect } from 'react';
import styles from '../styles/ModelTraining.module.css';
import Layout from '../components/Layout';
import api from '../lib/api';
import Select from 'react-select';

const allModels = [
  'LinearRegression',
  'RandomForestRegressor',
  'GradientBoostingRegressor',
  'SVR',
  'XGBRegressor',
  'LogisticRegression',
  'RandomForestClassifier',
  'GradientBoostingClassifier',
  'SVC',
  'XGBClassifier',
];
const allMetrics = [
  'RMSE', 'R2', 'MAE',
  'Accuracy', 'F1', 'Precision', 'Recall',
];

const customStyles = {
  control: (base) => ({
    ...base,
    backgroundColor: '#fff',
    borderColor: '#48C9B0',
    color: '#000',
  }),
  singleValue: (base) => ({ ...base, color: '#000' }),
  input: (base) => ({ ...base, color: '#000' }),
  placeholder: (base) => ({ ...base, color: '#000' }),
  menu: (base) => ({
    ...base,
    backgroundColor: '#fff',
    border: '1px solid #48C9B0',
  }),
  menuList: (base) => ({
    ...base,
    backgroundColor: '#fff',
  }),
  option: (base, state) => {
    let backgroundColor = '#fff';
    let color = '#000';
    if (state.isSelected) {
      backgroundColor = '#48C9B0';
      color = '#fff';
    } else if (state.isFocused) {
      backgroundColor = '#F0F0F0';
      color = '#000';
    }
    return { ...base, backgroundColor, color };
  },
};

export default function ModelTraining() {
  const [datasetSource, setDatasetSource] = useState('original');
  const [originalDatasetId, setOriginalDatasetId] = useState(null);
  const [preprocessedDatasetId, setPreprocessedDatasetId] = useState(null);
  const [featureEngineeredId, setFeatureEngineeredId] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [applyEncoding, setApplyEncoding] = useState(false);
  const [encodingMethod, setEncodingMethod] = useState('one-hot');
  const [applyScaling, setApplyScaling] = useState(false);
  const [scalingMethod, setScalingMethod] = useState('standard');
  const [columns, setColumns] = useState([]);
  const [loadingColumns, setLoadingColumns] = useState(false);
  const [targetOptions, setTargetOptions] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [featureOptions, setFeatureOptions] = useState([]);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [taskType, setTaskType] = useState('regression');
  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [enableCV, setEnableCV] = useState(false);
  const [cvFolds, setCvFolds] = useState(3);
  const [enableHyperTuning, setEnableHyperTuning] = useState(false);
  const [sampleData, setSampleData] = useState(false);
  const [sampleSize, setSampleSize] = useState(100000);
  const [hyperparameterSearchMethod, setHyperparameterSearchMethod] = useState("grid");
  const [randomSearchIter, setRandomSearchIter] = useState(10);
  const [trainingResults, setTrainingResults] = useState(null);
  const [loadingTrain, setLoadingTrain] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Use raw_dataset_id from preprocessing for the original dataset
  useEffect(() => {
    const origId = localStorage.getItem('raw_dataset_id');
    const prepId = localStorage.getItem('dataset_id');
    const feId = localStorage.getItem('featureEngineeredId');
    if (origId) setOriginalDatasetId(origId);
    if (prepId) setPreprocessedDatasetId(prepId);
    if (feId) setFeatureEngineeredId(feId);
  }, []);

  useEffect(() => {
    setColumns([]);
    setTargetOptions([]);
    setSelectedTarget(null);
    setFeatureOptions([]);
    setSelectedFeatures([]);
    setTrainingResults(null);
    if (datasetSource === 'preprocessed' && preprocessedDatasetId) {
      fetchColumns(preprocessedDatasetId, 'processed_data');
    } else if (datasetSource === 'feature_engineered' && featureEngineeredId) {
      fetchColumns(featureEngineeredId, 'feature_engineered_data');
    }
  }, [datasetSource, preprocessedDatasetId, featureEngineeredId]);

  useEffect(() => {
    if (selectedModels.length === 0) {
      setSelectedModels(getModelList());
    }
  }, [taskType]);

  async function fetchColumns(id, folder) {
    try {
      setLoadingColumns(true);
      const res = await api.get(`/feature-engineering/columns/?dataset_id=${id}&folder=${folder}`);
      if (res.data.status === 'success') {
        const cols = res.data.columns;
        setColumns(cols);
        const opts = cols.map((c) => ({ value: c, label: c }));
        setTargetOptions(opts);
        setFeatureOptions(opts);
      }
    } catch (err) {
      console.error(err);
      setErrorMessage(err.message);
    } finally {
      setLoadingColumns(false);
    }
  }

  function getModelList() {
    return taskType === 'regression'
      ? allModels.filter(m => m.toLowerCase().includes('regressor') || m.toLowerCase().includes('linear'))
      : allModels.filter(m =>
          m.toLowerCase().includes('classifier') ||
          m.toLowerCase().includes('logistic') ||
          m.toLowerCase().includes('svc')
        );
  }

  function getMetricList() {
    return taskType === 'regression'
      ? ['RMSE', 'R2', 'MAE']
      : ['Accuracy', 'F1', 'Precision', 'Recall'];
  }

  async function handleTrain() {
    setErrorMessage('');
    setTrainingResults(null);
    if (!selectedTarget) {
      setErrorMessage("Please select a target column.");
      return;
    }
    if (selectedModels.length === 0) {
      setErrorMessage("Please select at least one model.");
      return;
    }
    if (selectedMetrics.length === 0) {
      setErrorMessage("Please select at least one metric.");
      return;
    }
    try {
      setLoadingTrain(true);
      const params = {
        dataset_source: datasetSource,
        target_col: selectedTarget.value,
        task_type: taskType,
        selected_models: selectedModels,
        selected_metrics: selectedMetrics,
        selected_features: selectedFeatures.map(f => f.value),
        enable_cross_validation: enableCV,
        cv_folds: cvFolds,
        sample_data: sampleData,
        sample_size: sampleSize,
        hyperparameter_tuning: enableHyperTuning,
        hyperparameter_search_method: hyperparameterSearchMethod,
        random_search_iter: randomSearchIter,
        apply_encoding: applyEncoding,
        encoding_method: encodingMethod,
        apply_scaling: applyScaling,
        scaling_method: scalingMethod,
      };
      let formData = null;
      if (datasetSource === 'original') {
        // For original dataset, if a file is uploaded, use it.
        if (selectedFile) {
          formData = new FormData();
          formData.append('file', selectedFile);
        } else if (originalDatasetId) {
          params.dataset_id = originalDatasetId;
          params.folder = 'original_data';
        } else {
          setErrorMessage("Please upload a CSV file for the original dataset.");
          setLoadingTrain(false);
          return;
        }
      } else if (datasetSource === 'preprocessed') {
        if (!preprocessedDatasetId) {
          setErrorMessage("No preprocessed dataset ID found in localStorage.");
          setLoadingTrain(false);
          return;
        }
        params.dataset_id = preprocessedDatasetId;
      } else {
        if (!featureEngineeredId) {
          setErrorMessage("No featureEngineeredId found in localStorage.");
          setLoadingTrain(false);
          return;
        }
        params.feature_engineered_id = featureEngineeredId;
      }
      const response = await api.post('/model-training/train/', formData, {
        params,
        headers: formData ? { 'Content-Type': 'multipart/form-data' } : {},
      });
      setTrainingResults(response.data);
    } catch (err) {
      console.error(err);
      setErrorMessage(err.response?.data?.detail || err.message);
    } finally {
      setLoadingTrain(false);
    }
  }

  return (
    <Layout prevLink="/feature-engineering" nextLink="/forecasting">
      <div className={styles.container}>
        <h1 className={styles.title}>Model Training</h1>
        <p className={styles.description}>
          Select your dataset source, configure task parameters, and train your ML models.
        </p>
        <div className={styles.form}>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Dataset Source:</label>
            <select className={styles.select} value={datasetSource} onChange={(e) => setDatasetSource(e.target.value)}>
              <option value="original">Original Dataset</option>
              <option value="preprocessed">Preprocessed Dataset</option>
              <option value="feature_engineered">Feature-Engineered Dataset</option>
            </select>
          </div>
          {datasetSource === 'original' && (
            <div className={styles.inputGroup}>
              <label className={styles.label}>Upload CSV File (if new file needed):</label>
              <input type="file" className={styles.inputFile} onChange={(e) => setSelectedFile(e.target.files[0])} />
            </div>
          )}
          {datasetSource !== 'original' && (
            <div className={styles.inputGroup}>
              {loadingColumns && <p>Loading columns...</p>}
              {!loadingColumns && columns.length > 0 && <p>Fetched {columns.length} columns from server.</p>}
            </div>
          )}
        </div>
        <div className={styles.form}>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Task Type:</label>
            <select className={styles.select} value={taskType} onChange={(e) => setTaskType(e.target.value)}>
              <option value="regression">Regression</option>
              <option value="classification">Classification</option>
            </select>
          </div>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Target Column:</label>
            <Select
              className="reactSelectContainer"
              classNamePrefix="reactSelect"
              styles={customStyles}
              isClearable
              isSearchable
              placeholder="Select or search target..."
              options={targetOptions}
              value={selectedTarget}
              onChange={(val) => setSelectedTarget(val)}
            />
            <small>Type or pick from loaded columns if available.</small>
          </div>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Feature Columns:</label>
            <Select
              className="reactSelectContainer"
              classNamePrefix="reactSelect"
              styles={customStyles}
              isMulti
              isSearchable
              placeholder="Select or search features..."
              options={featureOptions}
              value={selectedFeatures}
              onChange={(vals) => setSelectedFeatures(vals)}
            />
            <small>Leave empty to use all columns except target.</small>
          </div>
        </div>
        <div className={styles.form}>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Models:</label>
            <div>
              {getModelList().map((m) => (
                <label key={m} style={{ marginRight: '1rem' }}>
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(m)}
                    onChange={() =>
                      selectedModels.includes(m)
                        ? setSelectedModels(selectedModels.filter(x => x !== m))
                        : setSelectedModels([...selectedModels, m])
                    }
                    style={{ marginRight: '4px' }}
                  />
                  {m}
                </label>
              ))}
            </div>
          </div>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Metrics:</label>
            <div>
              {getMetricList().map((met) => (
                <label key={met} style={{ marginRight: '1rem' }}>
                  <input
                    type="checkbox"
                    checked={selectedMetrics.includes(met)}
                    onChange={() =>
                      selectedMetrics.includes(met)
                        ? setSelectedMetrics(selectedMetrics.filter(x => x !== met))
                        : setSelectedMetrics([...selectedMetrics, met])
                    }
                    style={{ marginRight: '4px' }}
                  />
                  {met}
                </label>
              ))}
            </div>
          </div>
        </div>
        <div className={styles.form}>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Cross-Validation:</label>
            <label>
              <input
                type="checkbox"
                checked={enableCV}
                onChange={(e) => setEnableCV(e.target.checked)}
                style={{ marginRight: '4px' }}
              />
              Enable CV
            </label>
            {enableCV && (
              <div style={{ marginTop: '0.5rem' }}>
                <small>CV Folds:</small>
                <input
                  type="number"
                  min={2}
                  max={10}
                  value={cvFolds}
                  onChange={(e) => setCvFolds(Number(e.target.value))}
                  style={{ marginLeft: '0.5rem', width: '60px' }}
                />
              </div>
            )}
          </div>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Hyperparameter Tuning (GridSearch):</label>
            <label>
              <input
                type="checkbox"
                checked={enableHyperTuning}
                onChange={(e) => setEnableHyperTuning(e.target.checked)}
                style={{ marginRight: '4px' }}
              />
              Enable
            </label>
            {enableHyperTuning && (
              <div style={{ marginTop: '0.5rem' }}>
                <label style={{ marginRight: '1rem' }}>Search Method:</label>
                <select
                  value={hyperparameterSearchMethod}
                  onChange={(e) => setHyperparameterSearchMethod(e.target.value)}
                  style={{ marginRight: '1rem' }}
                >
                  <option value="grid">Grid Search</option>
                  <option value="random">Randomized Search</option>
                </select>
                {hyperparameterSearchMethod === "random" && (
                  <>
                    <label style={{ marginRight: '1rem' }}>Iterations:</label>
                    <input
                      type="number"
                      min={1}
                      value={randomSearchIter}
                      onChange={(e) => setRandomSearchIter(Number(e.target.value))}
                      style={{ width: '60px' }}
                    />
                  </>
                )}
              </div>
            )}
          </div>
          <div className={styles.inputGroup}>
            <label className={styles.label}>Sampling:</label>
            <label>
              <input
                type="checkbox"
                checked={sampleData}
                onChange={(e) => setSampleData(e.target.checked)}
                style={{ marginRight: '4px' }}
              />
              Sample Data?
            </label>
            {sampleData && (
              <div style={{ marginTop: '0.5rem' }}>
                <small>Sample Size:</small>
                <input
                  type="number"
                  min={1000}
                  max={1000000}
                  step={1000}
                  value={sampleSize}
                  onChange={(e) => setSampleSize(Number(e.target.value))}
                  style={{ marginLeft: '0.5rem', width: '80px' }}
                />
              </div>
            )}
          </div>
        </div>
        <div style={{ textAlign: 'center', marginTop: '1rem' }}>
          <button className={styles.button} onClick={handleTrain} disabled={loadingTrain}>
            {loadingTrain ? 'Training...' : 'Train Models'}
          </button>
          {errorMessage && <div className={styles.error}>{errorMessage}</div>}
        </div>
        {trainingResults && (
          <div className={styles.results}>
            <h2 className={styles.resultsTitle}>Training Results</h2>
            {trainingResults.status !== 'success' ? (
              <div className={styles.error}>{JSON.stringify(trainingResults)}</div>
            ) : (
              <>
                <div className={styles.tableContainer}>
                  <table className={styles.resultsTable}>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Training Time (s)</th>
                        {selectedMetrics.map(metric => (
                          <th key={metric}>{metric}</th>
                        ))}
                        {enableCV && <th>CV Score</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {trainingResults.results.map((result, index) => (
                        <tr key={index}>
                          <td>{result.Model}</td>
                          <td>{result.Training_Time_sec.toFixed(3)}</td>
                          {selectedMetrics.map(metric => (
                            <td
                              key={metric}
                              style={{
                                color: metric === 'R2' && result[metric] < 0 ? '#ff4444' : '#fff',
                                fontWeight: metric === 'R2' && result[metric] < 0 ? 'bold' : 'normal'
                              }}
                            >
                              {typeof result[metric] === 'number' ? result[metric].toFixed(4) : 'N/A'}
                            </td>
                          ))}
                          {enableCV && (
                            <td>
                              {result.CV_R2_Avg?.toFixed(4) ||
                               result.CV_F1_Avg?.toFixed(4) ||
                               'N/A'}
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className={styles.modelsSection}>
                  <h3>Trained Models</h3>
                  <table className={styles.modelsTable}>
                    <tbody>
                      {Object.entries(trainingResults.trained_models).map(([modelName, path]) => (
                        <tr key={modelName}>
                          <td className={styles.modelName}>{modelName}</td>
                          <td className={styles.modelPath}>{path}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </Layout>
  );
}

function getModelList() {
  return window.location.pathname.includes('regression')
    ? allModels.filter(m => m.toLowerCase().includes('regressor') || m.toLowerCase().includes('linear'))
    : allModels.filter(m =>
        m.toLowerCase().includes('classifier') ||
        m.toLowerCase().includes('logistic') ||
        m.toLowerCase().includes('svc')
      );
}

function getMetricList() {
  return window.location.pathname.includes('regression')
    ? ['RMSE', 'R2', 'MAE']
    : ['Accuracy', 'F1', 'Precision', 'Recall'];
}
