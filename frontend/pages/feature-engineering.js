import React, { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import api from '../lib/api';
import {
  Box,
  Button,
  Input,
  Text,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  FormControl,
  FormLabel,
  Select,
  Checkbox,
  NumberInput,
  NumberInputField,
  Spinner,
  Heading,
  useToast,
} from '@chakra-ui/react';

export default function FeatureEngineering() {
  // State for dataset selection
  const [datasetOption, setDatasetOption] = useState('original');

  const [selectedFile, setSelectedFile] = useState(null);
  const [originalPreview, setOriginalPreview] = useState(null);
  const [columns, setColumns] = useState([]);
  const [missingCounts, setMissingCounts] = useState({});
  const [numericCols, setNumericCols] = useState([]);
  const [objectCols, setObjectCols] = useState([]);
  const [dateCols, setDateCols] = useState([]);

  const [imputePlan, setImputePlan] = useState({});
  const [outlierPlan, setOutlierPlan] = useState({});
  const [dropOriginalDate, setDropOriginalDate] = useState({});
  const [convertPlan, setConvertPlan] = useState({});

  const [encodingMethod, setEncodingMethod] = useState('one-hot');
  const [scalingMethod, setScalingMethod] = useState('standard');
  const [applyPoly, setApplyPoly] = useState(false);
  const [polyDegree, setPolyDegree] = useState(2);
  const [interactionOnly, setInteractionOnly] = useState(false);
  const [includeBias, setIncludeBias] = useState(false);
  const [applyLog, setApplyLog] = useState(false);
  const [saveResult, setSaveResult] = useState(false);
  const [featureEngResult, setFeatureEngResult] = useState(null);
  const [tabIndex, setTabIndex] = useState(0);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const toast = useToast();
  const [downloadUrl, setDownloadUrl] = useState('');

  // New high cardinality options
  const [highCardOption, setHighCardOption] = useState('frequency');
  const [highCardThreshold, setHighCardThreshold] = useState(10);

  // For original dataset, stored id is under 'raw_dataset_id'
  const [storedOriginalId, setStoredOriginalId] = useState(null);

  useEffect(() => {
    const storedOrigId = localStorage.getItem('raw_dataset_id');
    if (storedOrigId) {
      setStoredOriginalId(storedOrigId);
    }
  }, []);

  useEffect(() => {
    if (datasetOption === 'preprocessed') {
      setTabIndex(4);
    }
  }, [datasetOption]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setOriginalPreview(null);
    setColumns([]);
    setMissingCounts({});
    setNumericCols([]);
    setObjectCols([]);
    setDateCols([]);
    setFeatureEngResult(null);

    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const lines = text.split('\n').filter((line) => line.trim() !== '');
      if (lines.length === 0) return;
      const headers = lines[0].split(',').map((h) => h.trim());
      const previewData = lines.slice(1, 11).map((line) => {
        const values = line.split(',').map((v) => v.trim());
        const rowObj = {};
        headers.forEach((header, i) => {
          rowObj[header] = values[i] || '';
        });
        return rowObj;
      });
      setOriginalPreview({ headers, data: previewData });
      setColumns(headers);
      // Optionally compute missing counts and infer column types here.
    };
    reader.readAsText(file);
  };

  const buildPayload = () => {
    const finalImputePlan = {};
    const finalOutlierPlan = {};
    const finalDatePlan = {};
    const finalConvertPlan = {};

    Object.keys(imputePlan).forEach((col) => {
      const val = imputePlan[col];
      if (val && val !== 'none') {
        finalImputePlan[col] = val;
      }
    });
    Object.keys(outlierPlan).forEach((col) => {
      const factor = outlierPlan[col];
      if (factor && factor > 0) {
        finalOutlierPlan[col] = factor;
      }
    });
    dateCols.forEach((col) => {
      finalDatePlan[col] = dropOriginalDate[col] ? true : false;
    });
    Object.keys(convertPlan).forEach((col) => {
      const ctype = convertPlan[col];
      if (ctype && ctype !== 'none') {
        finalConvertPlan[col] = ctype;
      }
    });
    return {
      impute_plan: JSON.stringify(finalImputePlan),
      outlier_plan: JSON.stringify(finalOutlierPlan),
      date_plan: JSON.stringify(finalDatePlan),
      convert_plan: JSON.stringify(finalConvertPlan),
    };
  };

  const handleRunFeatureEngineering = async () => {
    setError('');
    setLoading(true);
    // If the user selects preprocessed dataset, feature engineering is not required.
    if (datasetOption === 'preprocessed') {
      setError('Feature engineering is not required for preprocessed dataset.');
      setLoading(false);
      return;
    }
    try {
      const formData = new FormData();
      if (datasetOption === 'original') {
        const storedId = localStorage.getItem('raw_dataset_id');
        if (storedId) {
          formData.append('dataset_id', storedId);
          formData.append('folder', 'original_data');
        } else {
          if (!selectedFile) {
            setError('Please upload a file first.');
            setLoading(false);
            return;
          }
          formData.append('file', selectedFile);
        }
      }
      const payload = buildPayload();
      formData.append('impute_plan', payload.impute_plan);
      formData.append('outlier_plan', payload.outlier_plan);
      formData.append('date_plan', payload.date_plan);
      formData.append('convert_plan', payload.convert_plan);
      formData.append('encoding_method', encodingMethod);
      formData.append('scaling_method', scalingMethod);
      formData.append('apply_poly', applyPoly);
      formData.append('poly_degree', polyDegree);
      formData.append('interaction_only', interactionOnly);
      formData.append('include_bias', includeBias);
      formData.append('apply_log', applyLog);
      formData.append('save_result', saveResult);
      formData.append('high_card_threshold', highCardThreshold);
      formData.append('high_card_option', highCardOption);

      const response = await api.post('/feature-engineering/advanced/', formData);
      setFeatureEngResult(response.data);
      const blob = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
      toast({
        title: 'Feature Engineering Complete',
        description: `Shape: ${response.data.shape[0]} rows x ${response.data.shape[1]} columns`,
        status: 'success',
        duration: 4000,
        isClosable: true,
      });
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderPreviewTable = (data) => {
    if (!data || data.length === 0) return <Text>No data to display.</Text>;
    const cols = Object.keys(data[0]);
    return (
      <TableContainer border="1px" borderColor="gray.600" borderRadius="md" overflowX="auto">
        <Table variant="striped" colorScheme="gray" size="sm">
          <Thead bg="gray.700">
            <Tr>
              {cols.map((col) => (
                <Th key={col} color="white">{col}</Th>
              ))}
            </Tr>
          </Thead>
          <Tbody>
            {data.map((row, idx) => (
              <Tr key={idx}>
                {cols.map((col) => (
                  <Td key={col}>{row[col]}</Td>
                ))}
              </Tr>
            ))}
          </Tbody>
        </Table>
      </TableContainer>
    );
  };

  return (
    <Layout prevLink="/eda" nextLink="/model-training">
      <Heading as="h1" size="lg" mb={4}>🛠 Feature Engineering</Heading>
      <Box mb={4}>
        <Text mb={2}>Select the dataset for Feature Engineering:</Text>
        <FormControl>
          <Select value={datasetOption} onChange={(e) => setDatasetOption(e.target.value)}>
            <option value="original">Original Dataset</option>
            <option value="preprocessed">Preprocessed Dataset</option>
          </Select>
        </FormControl>
        {datasetOption === 'preprocessed' && (
          <Text mt={2} fontStyle="italic" color="gray.300">
            Using preprocessed dataset. Feature engineering is not applicable.
          </Text>
        )}
      </Box>
      <Tabs variant="soft-rounded" colorScheme="blue" index={tabIndex} onChange={(index) => setTabIndex(index)}>
        <TabList>
          <Tab>1. Upload</Tab>
          <Tab>2. Missing & Outliers</Tab>
          <Tab>3. Date & Convert</Tab>
          <Tab>4. Encoding/Scaling</Tab>
          <Tab>5. Preview & Run</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            {datasetOption === 'original' && !originalPreview && (
              <Box mb={4}>
                <Text mb={2}>Upload CSV File (no stored original dataset found):</Text>
                <Input type="file" onChange={handleFileChange} />
              </Box>
            )}
            {datasetOption === 'original' && originalPreview && (
              <>
                <Text fontWeight="bold" mb={2}>Original Data Preview (First 10 Rows)</Text>
                {renderPreviewTable(originalPreview.data)}
              </>
            )}
          </TabPanel>
          <TabPanel>
            <Heading as="h2" size="md" mb={4}>Missing Values & Outliers</Heading>
            {columns.length === 0 ? (
              <Text>No columns detected yet. Upload a file or wait for server fetch.</Text>
            ) : (
              <Box>
                {columns.map((col) => {
                  const colMissing = missingCounts[col] || 0;
                  return (
                    <Box key={col} mb={2}>
                      <Text fontWeight="bold">
                        {col} - Missing: {colMissing}
                      </Text>
                      <Select
                        placeholder="No action (none)"
                        onChange={(e) =>
                          setImputePlan((prev) => ({ ...prev, [col]: e.target.value }))
                        }
                      >
                        <option value="none">No action</option>
                        <option value="mean">Impute Mean</option>
                        <option value="median">Impute Median</option>
                        <option value="mode">Impute Mode</option>
                        <option value="knn">Impute KNN</option>
                        <option value="drop">Drop Missing Rows</option>
                      </Select>
                    </Box>
                  );
                })}
              </Box>
            )}
            <Box mt={6}>
              <Heading as="h2" size="md" mb={2}>Outlier Removal (Numeric Columns)</Heading>
              {numericCols.length === 0 ? (
                <Text>No numeric columns detected.</Text>
              ) : (
                <Box>
                  {numericCols.map((col) => (
                    <Box key={col} mb={2}>
                      <Text fontWeight="bold">{col}</Text>
                      <Text fontSize="sm">IQR Factor (0 = no removal)</Text>
                      <NumberInput
                        min={0}
                        max={5}
                        step={0.1}
                        defaultValue={0}
                        onChange={(valStr, valNum) =>
                          setOutlierPlan((prev) => ({ ...prev, [col]: valNum }))
                        }
                      >
                        <NumberInputField />
                      </NumberInput>
                    </Box>
                  ))}
                </Box>
              )}
            </Box>
          </TabPanel>
          <TabPanel>
            <Heading as="h2" size="md" mb={4}>Date & Convert</Heading>
            {columns.map((col) => (
              <Box key={col} mb={3}>
                <Checkbox
                  mr={2}
                  isChecked={dateCols.includes(col)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setDateCols((prev) => [...prev, col]);
                    } else {
                      setDateCols((prev) => prev.filter((c) => c !== col));
                    }
                  }}
                >
                  {col} is a date column?
                </Checkbox>
                {dateCols.includes(col) && (
                  <Box ml={6}>
                    <Checkbox
                      mr={2}
                      isChecked={!!dropOriginalDate[col]}
                      onChange={(e) =>
                        setDropOriginalDate((prev) => ({ ...prev, [col]: e.target.checked }))
                      }
                    >
                      Drop original?
                    </Checkbox>
                  </Box>
                )}
              </Box>
            ))}
            <Box mt={6}>
              <Heading as="h2" size="md" mb={2}>Convert Data Types</Heading>
              {columns.map((col) => (
                <Box key={col} mb={2}>
                  <Text fontWeight="bold">{col}</Text>
                  <Select
                    placeholder="No change (none)"
                    onChange={(e) =>
                      setConvertPlan((prev) => ({ ...prev, [col]: e.target.value }))
                    }
                  >
                    <option value="none">No change</option>
                    <option value="numeric">Convert to Numeric</option>
                    <option value="category">Convert to Category</option>
                  </Select>
                </Box>
              ))}
            </Box>
          </TabPanel>
          <TabPanel>
            <Heading as="h2" size="md" mb={4}>Global Transformations</Heading>
            <Box mb={4}>
              <FormLabel>Encoding Method</FormLabel>
              <Select
                value={encodingMethod}
                onChange={(e) => setEncodingMethod(e.target.value)}
              >
                <option value="one-hot">One-Hot</option>
                <option value="label">Label Encoding</option>
              </Select>
            </Box>
            <Box mb={4}>
              <FormLabel>Scaling Method</FormLabel>
              <Select
                value={scalingMethod}
                onChange={(e) => setScalingMethod(e.target.value)}
              >
                <option value="standard">StandardScaler</option>
                <option value="minmax">MinMaxScaler</option>
              </Select>
            </Box>
            <Box mb={4}>
              <Checkbox
                isChecked={applyPoly}
                onChange={(e) => setApplyPoly(e.target.checked)}
                mr={2}
              >
                Apply Polynomial Features?
              </Checkbox>
              {applyPoly && (
                <Box ml={6} mt={2}>
                  <Text>Polynomial Degree</Text>
                  <NumberInput
                    min={2}
                    max={5}
                    value={polyDegree}
                    onChange={(valStr, valNum) => setPolyDegree(valNum)}
                  >
                    <NumberInputField />
                  </NumberInput>
                  <Checkbox
                    isChecked={interactionOnly}
                    onChange={(e) => setInteractionOnly(e.target.checked)}
                    mr={2}
                    mt={2}
                  >
                    Interaction Only?
                  </Checkbox>
                  <br />
                  <Checkbox
                    isChecked={includeBias}
                    onChange={(e) => setIncludeBias(e.target.checked)}
                    mr={2}
                    mt={2}
                  >
                    Include Bias?
                  </Checkbox>
                </Box>
              )}
            </Box>
            <Box mb={4}>
              <Checkbox
                isChecked={applyLog}
                onChange={(e) => setApplyLog(e.target.checked)}
                mr={2}
              >
                Apply Log Transform?
              </Checkbox>
            </Box>
            <Box mb={4}>
              <Checkbox
                isChecked={saveResult}
                onChange={(e) => setSaveResult(e.target.checked)}
                mr={2}
              >
                Save Final CSV?
              </Checkbox>
            </Box>
          </TabPanel>
          <TabPanel>
            <Button colorScheme="blue" onClick={handleRunFeatureEngineering} mb={4}>
              Run Feature Engineering
            </Button>
            {loading && (
              <Box textAlign="center" my={4}>
                <Spinner size="xl" />
                <Text>Processing...</Text>
              </Box>
            )}
            {error && <Text color="red.500" mt={2}>{error}</Text>}
            {featureEngResult && (
              <Box mt={4}>
                <Text fontWeight="bold">
                  Feature-Engineered Data Preview (First 10 Rows)
                </Text>
                <Text>
                  Shape: {featureEngResult.shape[0]} x {featureEngResult.shape[1]}
                </Text>
                {featureEngResult.processed_file_path && (
                  <Text>
                    Saved at: <code>{featureEngResult.processed_file_path}</code>
                  </Text>
                )}
                {featureEngResult.sample_data
                  ? renderPreviewTable(featureEngResult.sample_data.slice(0, 10))
                  : <Text>No preview available.</Text>}
                {downloadUrl && (
                  <Button
                    as="a"
                    href={downloadUrl}
                    download="feature_engineered_data.json"
                    mt={4}
                    colorScheme="blue"
                  >
                    Download Feature Engineered Data
                  </Button>
                )}
              </Box>
            )}
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Layout>
  );
}
