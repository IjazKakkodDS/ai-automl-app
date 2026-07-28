import { useEffect, useState } from 'react';
import {
  Box,
  Heading,
  Text,
  Button,
  VStack,
  Container,
  List,
  ListItem,
  ListIcon,
  HStack
} from '@chakra-ui/react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRouter } from 'next/router';
import { CheckCircleIcon } from '@chakra-ui/icons';

export default function HeroSection() {
  const [isMounted, setIsMounted] = useState(false);
  const router = useRouter();

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Navigates to Preprocessing page
  const handleGetStarted = () => {
    router.push('/preprocessing');
  };

  // Smooth scroll to pipeline stages in index.js
  const handleExploreFeatures = () => {
    const pipelineSection = document.getElementById('pipeline-stages');
    if (pipelineSection) {
      pipelineSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <Box
      as="section"
      bgGradient="linear(to-b, gray.800, gray.900)"
      borderBottom="1px solid"
      borderColor="gray.700"
      py={{ base: 8, md: 12 }}
      px={4}
      textAlign="center"
    >
      <Container maxW="5xl">
        <AnimatePresence>
          {isMounted && (
            <VStack
              as={motion.div}
              spacing={{ base: 4, md: 6 }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.6 }}
            >
              {/* MAIN HEADLINE */}
              <Heading
                as="h1"
                fontSize={{ base: '3xl', md: '4xl', lg: '5xl' }}
                fontWeight="bold"
                color="teal.300"
              >
                AI AutoML Intelligence Platform
              </Heading>

              {/* SUBHEADING */}
              <Text
                fontSize={{ base: 'md', md: 'lg' }}
                color="gray.300"
                maxW="3xl"
                mx="auto"
              >
                A full-stack AutoML workflow system for dataset preprocessing,
                exploratory analysis, multi-model training, evaluation, and
                time-series forecasting. Each pipeline stage is a separately
                callable, inspectable API route backed by a dedicated agent
                module.
              </Text>

              {/* KEY HIGHLIGHTS */}
              <List
                spacing={2}
                color="gray.400"
                fontSize={{ base: 'sm', md: 'md' }}
                maxW="3xl"
                mx="auto"
                textAlign="left"
              >
                <ListItem>
                  <ListIcon as={CheckCircleIcon} color="teal.400" />
                  Automated Preprocessing, EDA &amp; Feature Engineering
                </ListItem>
                <ListItem>
                  <ListIcon as={CheckCircleIcon} color="teal.400" />
                  Multi-Model Training with SHAP Explainability
                </ListItem>
                <ListItem>
                  <ListIcon as={CheckCircleIcon} color="teal.400" />
                  Local FAISS Retrieval and Agentic AI Insights via Ollama
                </ListItem>
              </List>

              {/* CTA BUTTONS */}
              <HStack spacing={4}>
                <Button
                  size="md"
                  colorScheme="teal"
                  bg="teal.400"
                  _hover={{ bg: 'teal.300' }}
                  px={6}
                  py={4}
                  fontWeight={600}
                  onClick={handleGetStarted}
                >
                  Get Started
                </Button>
                <Button
                  size="md"
                  variant="outline"
                  colorScheme="teal"
                  borderColor="teal.400"
                  px={6}
                  py={4}
                  fontWeight={600}
                  onClick={handleExploreFeatures}
                >
                  Explore Features
                </Button>
              </HStack>

              {/* TAGLINE */}
              <Text fontSize="sm" color="gray.500" mt={1}>
                57 tests passed · Docker Compose orchestrated · benchmarked on
                synthetic and public tabular datasets.
              </Text>
            </VStack>
          )}
        </AnimatePresence>
      </Container>
    </Box>
  );
}
