import React, { useState } from 'react';
import {
  Box,
  Button,
  Input,
  Checkbox,
  NumberInput,
  NumberInputField,
  FormControl,
  FormLabel,
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
  Spinner,
  Select,
} from '@chakra-ui/react';
import Layout from '../components/Layout';
import api from '../lib/api';

export default function Preprocessing() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalPreview, setOriginalPreview] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState('');

  // Pipeline parameters
  const [imputeNumeric, setImputeNumeric] = useState('knn');
  const [knnNeighbors, setKnnNeighbors] = useState(3);
  const [removeOutliers, setRemoveOutliers] = useState(false);
  const [outlierFactor, setOutlierFactor] = useState(1.5);
  const [dropOriginalDateCols, setDropOriginalDateCols] = useState(true);
  const [dateDetectionThreshold, setDateDetectionThreshold] = useState(0.9);
  const [explicitDatetimeCols, setExplicitDatetimeCols] = useState('');

  // New high cardinality parameters
  const [highCardOption, setHighCardOption] = useState('frequency');
  const [highCardThreshold, setHighCardThreshold] = useState(10);

  // Simple CSV parser for local preview
  const parseCSV = (text) => {
    const lines = text.split('\n').filter((line) => line.trim() !== '');
    if (lines.length === 0) return { headers: [], data: [] };
    const headers = lines[0].split(',').map((h) => h.trim());
    const data = lines.slice(1).map((line) => {
      const values = line.split(',').map((val) => val.trim());
      const rowObj = {};
      headers.forEach((header, i) => {
        rowObj[header] = values[i] || '';
      });
      return rowObj;
    });
    return { headers, data };
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setOriginalPreview(null);
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const csvText = event.target.result;
      const parsed = parseCSV(csvText);
      setOriginalPreview(parsed.data.slice(0, 10)); // show first 10 rows
    };
    reader.readAsText(file);
  };

  const runPreprocessing = async () => {
    if (!selectedFile) {
      setError('Please select a CSV file first.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('impute_numeric', imputeNumeric);
      formData.append('knn_neighbors', knnNeighbors);
      formData.append('remove_outliers', removeOutliers);
      formData.append('outlier_factor', outlierFactor);
      formData.append('drop_original_date_cols', dropOriginalDateCols);
      formData.append('date_detection_threshold', dateDetectionThreshold);
      formData.append('explicit_datetime_cols', explicitDatetimeCols);
      formData.append('high_card_threshold', highCardThreshold);
      formData.append('high_card_option', highCardOption);

      const response = await api.post('/preprocessing/preprocess/', formData);
      setProcessedData(response.data);

      // Store the returned IDs in localStorage so that the EDA stage can pick them up.
      localStorage.setItem('raw_dataset_id', response.data.raw_dataset_id);
      localStorage.setItem('dataset_id', response.data.dataset_id);

      // Create a download URL for the processed dataset (as JSON)
      const blob = new Blob(
        [JSON.stringify(response.data, null, 2)],
        { type: 'application/json' }
      );
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderTable = (data, title) => {
    if (!data || data.length === 0) return <Text>No preview available.</Text>;
    const cols = Object.keys(data[0]);
    return (
      <Box mb={4}>
        <Text fontWeight="bold" mb={2}>{title}</Text>
        <TableContainer border="1px" borderColor="gray.600" borderRadius="md" overflowX="auto">
          <Table variant="striped" size="sm">
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
      </Box>
    );
  };

  return (
    <Layout prevLink="/home" nextLink="/eda">
      <Box p={4}>
        <Text fontSize="2xl" fontWeight="bold" mb={4}>
          Data Preprocessing
        </Text>

        {/* File Upload */}
        <FormControl mb={4}>
          <FormLabel>Upload CSV File</FormLabel>
          <Input type="file" onChange={handleFileChange} />
        </FormControl>

        {/* Numeric Imputation Options */}
        <FormControl mb={4}>
          <FormLabel>Numeric Imputation Strategy</FormLabel>
          <Select value={imputeNumeric} onChange={(e) => setImputeNumeric(e.target.value)}>
            <option value="knn">KNN</option>
            <option value="mean">Mean</option>
            <option value="median">Median</option>
          </Select>
        </FormControl>

        {imputeNumeric === 'knn' && (
          <FormControl mb={4}>
            <FormLabel>KNN Neighbors</FormLabel>
            <NumberInput
              min={1}
              max={20}
              step={1}
              value={knnNeighbors}
              onChange={(valueStr, valueNum) => setKnnNeighbors(valueNum)}
            >
              <NumberInputField />
            </NumberInput>
          </FormControl>
        )}

        {/* Outlier Removal Option */}
        <FormControl display="flex" alignItems="center" mb={4}>
          <Checkbox
            isChecked={removeOutliers}
            onChange={(e) => setRemoveOutliers(e.target.checked)}
            mr={2}
          >
            Remove Outliers (IQR)
          </Checkbox>
        </FormControl>

        {removeOutliers && (
          <FormControl mb={4}>
            <FormLabel>Outlier Factor (IQR multiplier)</FormLabel>
            <NumberInput
              min={1}
              max={5}
              step={0.1}
              value={outlierFactor}
              onChange={(valueStr, valueNum) => setOutlierFactor(valueNum)}
            >
              <NumberInputField />
            </NumberInput>
          </FormControl>
        )}

        {/* Date Options */}
        <FormControl display="flex" alignItems="center" mb={4}>
          <Checkbox
            isChecked={dropOriginalDateCols}
            onChange={(e) => setDropOriginalDateCols(e.target.checked)}
            mr={2}
          >
            Drop Original Date Columns
          </Checkbox>
        </FormControl>

        <FormControl mb={4}>
          <FormLabel>Date Detection Threshold</FormLabel>
          <NumberInput
            min={0}
            max={1}
            step={0.05}
            value={dateDetectionThreshold}
            onChange={(valueStr, valueNum) => setDateDetectionThreshold(valueNum)}
          >
            <NumberInputField />
          </NumberInput>
          <Text fontSize="sm" color="gray.400">
            Proportion of valid dates required to treat a column as datetime.
          </Text>
        </FormControl>

        <FormControl mb={4}>
          <FormLabel>Explicit Datetime Columns (comma-separated)</FormLabel>
          <Input
            placeholder="e.g. DateCol,AnotherDate"
            value={explicitDatetimeCols}
            onChange={(e) => setExplicitDatetimeCols(e.target.value)}
          />
        </FormControl>

        {/* High Cardinality Options */}
        <FormControl mb={4}>
          <FormLabel>High Cardinality Encoding Option</FormLabel>
          <Select value={highCardOption} onChange={(e) => setHighCardOption(e.target.value)}>
            <option value="one-hot">One-Hot Encoding</option>
            <option value="frequency">Frequency Encoding</option>
            <option value="drop">Drop Variable</option>
          </Select>
        </FormControl>

        <FormControl mb={4}>
          <FormLabel>High Cardinality Threshold</FormLabel>
          <NumberInput
            min={1}
            max={100}
            step={1}
            value={highCardThreshold}
            onChange={(valueStr, valueNum) => setHighCardThreshold(valueNum)}
          >
            <NumberInputField />
          </NumberInput>
          <Text fontSize="sm" color="gray.400">
            Maximum unique values for one-hot encoding. Variables above this threshold will use the selected strategy.
          </Text>
        </FormControl>

        <Button onClick={runPreprocessing} colorScheme="blue" isLoading={loading}>
          Run Preprocessing
        </Button>

        {error && <Text color="red.500" mt={4}>{error}</Text>}

        {/* Tabs for previewing Original and Preprocessed Data */}
        {selectedFile && (
          <Tabs variant="soft-rounded" colorScheme="blue" mt={4}>
            <TabList>
              <Tab>Original Data Preview</Tab>
              <Tab>Preprocessed Data Preview</Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                {originalPreview
                  ? renderTable(originalPreview, 'Original Dataset (First 10 Rows)')
                  : <Text>No preview available.</Text>}
              </TabPanel>
              <TabPanel>
                {processedData ? (
                  <>
                    <Text>
                      Data Shape: {processedData.shape[0]} rows x {processedData.shape[1]} columns
                    </Text>
                    {processedData.sample_data
                      ? renderTable(processedData.sample_data, 'Preprocessed Dataset (Sample)')
                      : <Text>No preview available.</Text>}
                    {downloadUrl && (
                      <Button
                        as="a"
                        href={downloadUrl}
                        download="preprocessed_data.json"
                        mt={4}
                        colorScheme="blue"
                      >
                        Download Preprocessed Data
                      </Button>
                    )}
                  </>
                ) : (
                  <Text>No processed data to display yet. Run preprocessing first.</Text>
                )}
              </TabPanel>
            </TabPanels>
          </Tabs>
        )}
        {loading && (
          <Box textAlign="center" my={4}>
            <Spinner size="xl" />
            <Text>Processing...</Text>
          </Box>
        )}
      </Box>
    </Layout>
  );
}
