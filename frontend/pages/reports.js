import React, { useState } from 'react';
import Layout from '../components/Layout';
import api from '../lib/api';
import { Box, Button, Flex, Text, useToast } from '@chakra-ui/react';
import styles from '../styles/Reports.module.css';

export default function Reports() {
  const [reportStatus, setReportStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const toast = useToast();
  const [reportUrl, setReportUrl] = useState('');

  // Generate full report (requires dataset_id in localStorage)
  const generateReport = async () => {
    setLoading(true);
    setReportStatus('Generating report...');
    try {
      const datasetId = localStorage.getItem('dataset_id');
      if (!datasetId) {
        setReportStatus('Error: No dataset_id found in localStorage.');
        setLoading(false);
        return;
      }

      await api.post('/reports/generate-full', null, {
        params: { dataset_id: datasetId },
      });

      setReportStatus('Report generated! You can now preview or download it.');
      toast({
        title: 'Report Generated',
        description: 'You can now preview or download the report.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      setReportUrl('http://127.0.0.1:8000/reports/view-full'); // or your actual server address
    } catch (err) {
      console.error(err);
      setReportStatus('Error generating report.');
      toast({
        title: 'Error',
        description: 'Error generating report.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  // Download the full PDF
  const downloadReport = () => {
    window.open('http://127.0.0.1:8000/reports/download-full', '_blank');
  };

  // Preview the HTML report
  const previewReport = () => {
    if (!reportUrl) {
      setReportStatus('Please generate the report first.');
      return;
    }
    window.open(reportUrl, '_blank');
  };

  return (
    <Layout prevLink="/rag" nextLink="/">
      <Box className={styles.container}>
        <Text className={styles.title} fontSize="2xl" mb={4}>
          Comprehensive Reports & Downloads
        </Text>

        <Button onClick={generateReport} isLoading={loading} colorScheme="teal" mb={4}>
          Generate Full Report
        </Button>

        {reportStatus && <Text className={styles.status} mb={4}>{reportStatus}</Text>}

        <Flex gap={4} mb={4}>
          <Button onClick={previewReport} colorScheme="teal" isLoading={loading}>
            Preview Report
          </Button>
          <Button onClick={downloadReport} colorScheme="teal">
            Download PDF Report
          </Button>
        </Flex>
      </Box>
    </Layout>
  );
}
