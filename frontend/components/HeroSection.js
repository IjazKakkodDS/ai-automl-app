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
                Enterprise-Grade AutoML Platform
              </Heading>

              {/* SUBHEADING */}
              <Text
                fontSize={{ base: 'md', md: 'lg' }}
                color="gray.300"
                maxW="3xl"
                mx="auto"
              >
                Empower your teams with advanced automation—from data ingestion
                to production deployment. Integrate retrieval-augmented insights
                and agentic AI for faster iteration, greater transparency, and
                stronger decision-making.
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
                  Automated EDA, Feature Engineering &amp; Model Selection
                </ListItem>
                <ListItem>
                  <ListIcon as={CheckCircleIcon} color="teal.400" />
                  RAG-Based AI Guidance from Documentation &amp; Logs
                </ListItem>
                <ListItem>
                  <ListIcon as={CheckCircleIcon} color="teal.400" />
                  Explainable, End-to-End Workflows for Confident Deployments
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
                Perfect for analytics managers &amp; data scientists seeking
                faster, more impactful insights.
              </Text>
            </VStack>
          )}
        </AnimatePresence>
      </Container>
    </Box>
  );
}
