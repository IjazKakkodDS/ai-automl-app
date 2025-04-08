import React, { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import api from '../lib/api';
import {
  Box,
  Button,
  Input,
  Checkbox,
  FormControl,
  FormLabel,
  Text,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Spinner,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  Select,
  Radio,
  RadioGroup,
  Stack,
  NumberInput,
  NumberInputField,
} from '@chakra-ui/react';
import dynamic from 'next/dynamic';

// Dynamically import Plotly for interactive charts
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function EDA() {
  // User selects between original and preprocessed datasets.
  const [datasetOption, setDatasetOption] = useState('original'); // 'original' or 'preprocessed'
  const [selectedFile, setSelectedFile] = useState(null);
  const [excludeDates, setExcludeDates] = useState(true);
  const [interactive, setInteractive] = useState(true);
  const [correlationMethod, setCorrelationMethod] = useState('pearson');
  const [sampleSize, setSampleSize] = useState(100000);
  const [maxNumericCols, setMaxNumericCols] = useState(8);
  const [edaResult, setEdaResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedStaticPlot, setSelectedStaticPlot] = useState('');

  // Stored dataset IDs from the preprocessing stage.
  const [storedOriginalId, setStoredOriginalId] = useState(null);
  const [storedProcessedId, setStoredProcessedId] = useState(null);

  useEffect(() => {
    // On mount, read stored IDs from localStorage.
    // For the original dataset, use the key "raw_dataset_id"
    setStoredOriginalId(localStorage.getItem('raw_dataset_id'));
    setStoredProcessedId(localStorage.getItem('dataset_id'));
  }, []);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    // If a new file is provided, clear the stored IDs.
    if (datasetOption === 'original') {
      localStorage.removeItem('raw_dataset_id');
      setStoredOriginalId(null);
    } else {
      localStorage.removeItem('dataset_id');
      setStoredProcessedId(null);
    }
  };

  const runEDA = async () => {
    setError('');
    setLoading(true);
    setEdaResult(null);
    setSelectedStaticPlot('');
    try {
      const formData = new FormData();
      // If a new file is provided, use that.
      if (selectedFile) {
        formData.append('file', selectedFile);
      } else {
        // Otherwise, use the stored dataset ID based on the selected option.
        const storageKey = datasetOption === 'original' ? 'raw_dataset_id' : 'dataset_id';
        const folderName = datasetOption === 'original' ? 'original_data' : 'processed_data';
        const id = localStorage.getItem(storageKey);
        if (!id) {
          setError('No dataset found. Please upload your CSV file in the preprocessing stage.');
          setLoading(false);
          return;
        }
        formData.append('dataset_id', id);
        formData.append('folder', folderName);
      }
      // Append additional EDA parameters
      formData.append('exclude_date_features', excludeDates);
      formData.append('interactive', interactive);
      formData.append('correlation_method', correlationMethod);
      formData.append('sample_size', sampleSize.toString());
      formData.append('max_numeric_cols', maxNumericCols.toString());
      const response = await api.post('/eda/analysis/', formData);
      setEdaResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderTable = (tableData, title) => {
    if (!tableData || tableData.length === 0) return null;
    const columns = Object.keys(tableData[0]);
    return (
      <Box mb={4}>
        <Text fontWeight="bold" mb={2}>{title}</Text>
        <TableContainer border="1px" borderColor="gray.600" borderRadius="md" overflowX="auto">
          <Table variant="striped" size="sm">
            <Thead bg="gray.700">
              <Tr>
                {columns.map((col) => (
                  <Th key={col} color="white">{col}</Th>
                ))}
              </Tr>
            </Thead>
            <Tbody>
              {tableData.map((row, idx) => (
                <Tr key={idx}>
                  {columns.map((col, colIdx) => (
                    <Td key={colIdx}>{row[col]}</Td>
                  ))}
                </Tr>
              ))}
            </Tbody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  const renderFigure = (figStr, title) => {
    if (!figStr) return null;
    if (figStr.startsWith('data:image/png;base64,')) {
      return (
        <Box mb={4}>
          <Text fontWeight="bold" mb={2}>{title}</Text>
          <img src={figStr} alt={title} style={{ width: '100%' }} />
        </Box>
      );
    } else {
      try {
        const figJson = JSON.parse(figStr);
        return (
          <Box mb={4}>
            <Text fontWeight="bold" mb={2}>{title}</Text>
            <Plot
              data={figJson.data}
              layout={figJson.layout}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </Box>
        );
      } catch (err) {
        return <Text color="red.300">Failed to load Plotly figure: {err.message}</Text>;
      }
    }
  };

  const staticPlotKeys = edaResult?.figures
    ? Object.keys(edaResult.figures).filter((key) => {
        const val = edaResult.figures[key];
        return val.startsWith('data:image/png;base64,');
      })
    : [];

  return (
    <Layout prevLink="/preprocessing" nextLink="/feature-engineering">
      <Box color="white" maxW="900px" mx="auto" p={4}>
        <Text fontSize="2xl" fontWeight="bold" mb={4}>Exploratory Data Analysis (EDA)</Text>
        <Box mb={4}>
          <Text mb={2}>Select dataset source:</Text>
          <RadioGroup onChange={setDatasetOption} value={datasetOption} mb={4}>
            <Stack direction="row">
              <Radio value="original">Original Dataset</Radio>
              <Radio value="preprocessed">Preprocessed Dataset</Radio>
            </Stack>
          </RadioGroup>
          {/* Display upload option only if a stored ID is not present */}
          {datasetOption === 'original' && !storedOriginalId && (
            <>
              <Text fontSize="sm" color="gray.400" mb={1}>
                Upload your CSV file once in the preprocessing stage. No need to reupload for EDA.
              </Text>
              <Input type="file" onChange={handleFileChange} mb={3} />
            </>
          )}
          {datasetOption === 'preprocessed' && !storedProcessedId && (
            <>
              <Text fontSize="sm" color="gray.400" mb={1}>
                Upload your CSV file once in the preprocessing stage. No need to reupload for EDA.
              </Text>
              <Input type="file" onChange={handleFileChange} mb={3} />
            </>
          )}
          {/* Show a message if the file is already uploaded */}
          {datasetOption === 'original' && storedOriginalId && (
            <Text fontSize="sm" color="green.300" mb={3}>
              Original dataset found. (ID: {storedOriginalId})
            </Text>
          )}
          {datasetOption === 'preprocessed' && storedProcessedId && (
            <Text fontSize="sm" color="green.300" mb={3}>
              Preprocessed dataset found. (ID: {storedProcessedId})
            </Text>
          )}
          <FormControl display="flex" alignItems="center" mb={2}>
            <Checkbox isChecked={excludeDates} onChange={(e) => setExcludeDates(e.target.checked)} mr={2}>
              Exclude Date-Derived Columns
            </Checkbox>
          </FormControl>
          <FormControl display="flex" alignItems="center" mb={2}>
            <Checkbox isChecked={interactive} onChange={(e) => setInteractive(e.target.checked)} mr={2}>
              Include Interactive Plots (Plotly)
            </Checkbox>
          </FormControl>
          <FormControl mb={2}>
            <FormLabel>Correlation Method</FormLabel>
            <Select value={correlationMethod} onChange={(e) => setCorrelationMethod(e.target.value)}>
              <option value="pearson">Pearson</option>
              <option value="spearman">Spearman</option>
              <option value="kendall">Kendall</option>
            </Select>
          </FormControl>
          <FormControl mb={2}>
            <FormLabel>Sample Size for EDA (For very large datasets)</FormLabel>
            <NumberInput min={1000} max={1000000} step={5000} value={sampleSize} onChange={(valStr, valNum) => setSampleSize(valNum)}>
              <NumberInputField />
            </NumberInput>
          </FormControl>
          <FormControl mb={4}>
            <FormLabel>Max Numeric Columns (for pairplot)</FormLabel>
            <NumberInput min={2} max={20} step={1} value={maxNumericCols} onChange={(valStr, valNum) => setMaxNumericCols(valNum)}>
              <NumberInputField />
            </NumberInput>
          </FormControl>
          <Button onClick={runEDA} colorScheme="teal" mb={2}>Run EDA</Button>
        </Box>
        {loading && (
          <Box textAlign="center" my={4}>
            <Spinner size="xl" />
            <Text mt={2}>Processing EDA...</Text>
          </Box>
        )}
        {error && <Text color="red.400">{error}</Text>}
        {edaResult && (
          <Tabs variant="soft-rounded" colorScheme="teal" mt={4}>
            <TabList>
              <Tab>Report</Tab>
              <Tab>Summary Tables</Tab>
              <Tab>All Figures</Tab>
              <Tab>Static Plots</Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                <Text whiteSpace="pre-wrap" fontFamily="monospace">{edaResult.eda_report}</Text>
              </TabPanel>
              <TabPanel>
                {edaResult.tables && Object.keys(edaResult.tables).map((tableName) => (
                  <Box key={tableName} mb={6}>{renderTable(edaResult.tables[tableName], tableName)}</Box>
                ))}
              </TabPanel>
              <TabPanel>
                {edaResult.figures && Object.keys(edaResult.figures).map((figName) => (
                  <Box key={figName} mb={6}>{renderFigure(edaResult.figures[figName], figName)}</Box>
                ))}
              </TabPanel>
              <TabPanel>
                <FormControl mb={4}>
                  <FormLabel>Select a Static Plot to Display:</FormLabel>
                  <Select placeholder="Choose a plot" onChange={(e) => setSelectedStaticPlot(e.target.value)} value={selectedStaticPlot}>
                    {staticPlotKeys.map((key) => (
                      <option key={key} value={key}>{key}</option>
                    ))}
                  </Select>
                </FormControl>
                {selectedStaticPlot && renderFigure(edaResult.figures[selectedStaticPlot], selectedStaticPlot)}
              </TabPanel>
            </TabPanels>
          </Tabs>
        )}
      </Box>
    </Layout>
  );
}
