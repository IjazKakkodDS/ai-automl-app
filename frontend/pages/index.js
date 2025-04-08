import { useEffect, useState } from 'react';
import {
  Box,
  VStack,
  Heading,
  Text,
  SimpleGrid,
  Icon,
  Flex,
  useColorModeValue,
  Container,
  Badge,
  Divider
} from '@chakra-ui/react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import Layout from '../components/Layout';
import HeroSection from '../components/HeroSection';

// Recharts imports
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell
} from 'recharts';

// Icons
import {
  FiArrowUpRight,
  FiDatabase,
  FiCode,
  FiZap,
  FiActivity,
  FiTrendingUp,
  FiStar,
  FiInfo,
  FiFileText
} from 'react-icons/fi';
import { InfoIcon } from '@chakra-ui/icons';

// 1) Sample area chart data (training metrics)
const trainingData = [
  { epoch: 1, accuracy: 0.72, loss: 0.85 },
  { epoch: 2, accuracy: 0.76, loss: 0.66 },
  { epoch: 3, accuracy: 0.80, loss: 0.55 },
  { epoch: 4, accuracy: 0.83, loss: 0.47 },
  { epoch: 5, accuracy: 0.86, loss: 0.41 },
  { epoch: 6, accuracy: 0.88, loss: 0.36 }
];

// 2) Sample pie chart data (class distribution)
const classDistData = [
  { label: 'Class A', value: 420 },
  { label: 'Class B', value: 380 },
  { label: 'Class C', value: 200 },
  { label: 'Class D', value: 120 }
];
const pieColors = ['#00B4D8', '#FFD700', '#EE7752', '#00FF88'];

// 3) Pipeline items (matching your sidebar links)
const pipelineStages = [
  {
    label: 'Preprocessing',
    href: '/preprocessing',
    description: 'Data cleaning, transformation & validation',
    icon: FiDatabase,
    badge: 'Core'
  },
  {
    label: 'EDA',
    href: '/eda',
    description: 'Exploratory data analysis & insights',
    icon: FiActivity,
    badge: 'Core'
  },
  {
    label: 'Feature Eng',
    href: '/feature-engineering',
    description: 'Feature generation & selection',
    icon: FiCode,
    badge: 'Advanced'
  },
  {
    label: 'Model Training',
    href: '/model-training',
    description: 'Train models with flexible hyperparameters',
    icon: FiZap,
    badge: 'Core'
  },
  {
    label: 'Forecasting',
    href: '/forecasting',
    description: 'Time-series forecasting & predictions',
    icon: FiTrendingUp,
    badge: 'Core'
  },
  {
    label: 'Evaluate',
    href: '/evaluate',
    description: 'Assess model performance & metrics',
    icon: FiStar,
    badge: 'Advanced'
  },
  {
    label: 'AI Insights',
    href: '/insights',
    description: 'Interpretability and deeper AI insights',
    icon: FiInfo,
    badge: 'Core'
  },
  {
    label: 'RAG & Agentic AI',
    href: '/rag',
    description: 'Retrieval-Augmented Generation & Agentic AI',
    icon: InfoIcon,
    badge: 'Advanced'
  },
  {
    label: 'Reports',
    href: '/reports',
    description: 'Generate & share comprehensive reports',
    icon: FiFileText,
    badge: 'Core'
  }
];

export default function Home() {
  const [isMounted, setIsMounted] = useState(false);
  const cardBg = useColorModeValue('gray.800', 'gray.700');
  const borderColor = useColorModeValue('gray.100', 'gray.600');

  useEffect(() => {
    setIsMounted(true);
  }, []);

  return (
    <Layout>
      {/* 
        HeroSection:
        The "Explore Features" button scrolls to #pipeline-stages
      */}
      <HeroSection />

      {/* INTRODUCTORY SECTION */}
      <Container maxW="7xl" py={8}>
        <Heading as="h2" size="md" color="teal.300" mb={4}>
          Production Architecture & Workflows
        </Heading>

        <Box
          as={motion.div}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <Text color="gray.300" mb={3}>
            This application addresses the most time-consuming steps in
            the machine learning lifecycle—data ingestion, automated EDA,
            and model building—while offering deep explainability through
            Retrieval-Augmented Generation (RAG) and agentic AI.
          </Text>
          <Text color="gray.400" fontSize="sm">
            By unifying these workflows within one platform, users can quickly
            upload datasets, run forecasts, generate robust metrics, and
            deploy models with minimal friction. The result is faster
            iteration, consistent best practices, and transparent,
            data-driven decisions—ideal for both prototypes and production.
          </Text>
        </Box>
      </Container>

      <Divider my={8} borderColor="gray.600" />

      {/* MODEL PERFORMANCE & CLASS DISTRIBUTION */}
      <Container maxW="7xl" py={4}>
        <Heading as="h2" size="md" color="teal.300" mb={4}>
          Model Performance & Distribution
        </Heading>

        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
          {/* Chart #1: Accuracy & Loss (AreaChart) */}
          <Box
            bg="gray.900"
            borderRadius="md"
            border="1px solid"
            borderColor="gray.700"
            p={4}
            height="380px"
            display="flex"
            flexDirection="column"
          >
            <Text color="gray.200" fontWeight="bold" mb={2}>
              Training Metrics
            </Text>
            {isMounted ? (
              <Box flex="1">
                <ResponsiveContainer width="100%" height="80%">
                  <AreaChart
                    data={trainingData}
                    margin={{ top: 20, right: 20, left: 0, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00FF88" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#00FF88" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00B4D8" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#00B4D8" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="gray" />
                    <XAxis dataKey="epoch" stroke="#ccc" />
                    <YAxis stroke="#ccc" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1A202C', border: 'none' }}
                      labelStyle={{ color: '#fff' }}
                      itemStyle={{ color: '#fff' }}
                    />
                    <Legend wrapperStyle={{ color: '#fff' }} />
                    <Area
                      type="monotone"
                      dataKey="accuracy"
                      stroke="#00FF88"
                      fillOpacity={1}
                      fill="url(#colorAcc)"
                    />
                    <Area
                      type="monotone"
                      dataKey="loss"
                      stroke="#00B4D8"
                      fillOpacity={1}
                      fill="url(#colorLoss)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            ) : (
              <Text color="gray.500">Loading chart...</Text>
            )}

            <Text color="gray.400" fontSize="sm" mt={2}>
              By epoch 6, <strong>accuracy</strong> reached <strong>88%</strong>,
              while <strong>loss</strong> dropped to <strong>0.36</strong>.
            </Text>
          </Box>

          {/* Chart #2: Class Distribution (PieChart) */}
          <Box
            bg="gray.900"
            borderRadius="md"
            border="1px solid"
            borderColor="gray.700"
            p={4}
            height="400px"
            display="flex"
            flexDirection="column"
          >
            <Text color="gray.200" fontWeight="bold" mb={2}>
              Class Distribution
            </Text>
            {isMounted ? (
              <Box flex="1">
                <ResponsiveContainer width="100%" height="80%">
                  <PieChart>
                    <Pie
                      data={classDistData}
                      dataKey="value"
                      nameKey="label"
                      outerRadius={90}
                      innerRadius={50}
                      label={false}
                      labelLine={false}
                    >
                      {classDistData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={pieColors[index % pieColors.length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1A202C', border: 'none' }}
                      labelStyle={{ color: '#fff' }}
                      itemStyle={{ color: '#fff' }}
                    />
                    <Legend
                      verticalAlign="bottom"
                      align="center"
                      iconSize={14}
                      wrapperStyle={{
                        color: '#fff',
                        marginTop: '14px',
                        fontSize: '0.9rem'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            ) : (
              <Text color="gray.500">Loading chart...</Text>
            )}

            <Text color="gray.400" fontSize="sm" mt={2}>
              <strong>Class A</strong> has the most samples (<strong>420</strong>),
              while <strong>Class D</strong> is smallest at <strong>120</strong>.
            </Text>
          </Box>
        </SimpleGrid>
      </Container>

      <Divider my={8} borderColor="gray.600" />

      {/* KEY PIPELINE STAGES */}
      <Container maxW="7xl" py={4} id="pipeline-stages">
        <Heading as="h2" size="md" color="teal.300" mb={2}>
          Key Pipeline Stages
        </Heading>
        <Text color="gray.400" mb={4}>
          From data ingestion to deployment, each stage in our pipeline
          contributes to reliable, explainable models.
        </Text>

        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
          {pipelineStages.map((stage) => (
            <motion.div
              key={stage.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              viewport={{ once: true }}
            >
              <Link href={stage.href}>
                <Box
                  p={6}
                  bg={cardBg}
                  borderRadius="md"
                  border="1px solid"
                  borderColor={borderColor}
                  _hover={{
                    borderColor: stage.badge === 'Core' ? 'teal.300' : 'blue.300',
                    transform: 'translateY(-2px)'
                  }}
                  transition="all 0.2s ease"
                  cursor="pointer"
                  minH="180px"
                  display="flex"
                  flexDirection="column"
                  justifyContent="space-between"
                >
                  <Flex align="center" gap={3} mb={3}>
                    <Box
                      p={3}
                      bg={
                        stage.badge === 'Core'
                          ? 'rgba(56, 178, 172, 0.1)'
                          : 'rgba(66, 153, 225, 0.1)'
                      }
                      borderRadius="md"
                    >
                      <Icon
                        as={stage.icon}
                        boxSize={7}
                        color={stage.badge === 'Core' ? 'teal.300' : 'blue.300'}
                      />
                    </Box>

                    <Box>
                      <Flex align="center" gap={2}>
                        <Text fontWeight="bold" fontSize="lg" color="gray.200">
                          {stage.label}
                        </Text>
                        <Badge
                          variant="subtle"
                          colorScheme={
                            stage.badge.toLowerCase() === 'core' ? 'teal' : 'blue'
                          }
                          fontSize="0.7rem"
                          px={2}
                          py={1}
                        >
                          {stage.badge}
                        </Badge>
                      </Flex>
                      <Text fontSize="sm" color="gray.400" mt={1}>
                        {stage.description}
                      </Text>
                    </Box>
                  </Flex>

                  <Flex align="center" justify="flex-end">
                    <Icon as={FiArrowUpRight} boxSize={5} color="gray.500" />
                  </Flex>
                </Box>
              </Link>
            </motion.div>
          ))}
        </SimpleGrid>
      </Container>
    </Layout>
  );
}
