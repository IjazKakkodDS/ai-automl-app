import React, { useState } from 'react';
import { Flex, Box, Link as ChakraLink, Button, HStack, useColorModeValue } from '@chakra-ui/react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import Sidebar from './Sidebar';
import Navbar from './Navbar';

export default function Layout({ children, prevLink, nextLink }) {
  const router = useRouter();
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Subtle adjustments for color and borders
  const layoutBg = useColorModeValue('gray.900', 'gray.900');
  const contentBoxShadow = useColorModeValue('inset 0 0 6px rgba(0, 0, 0, 0.3)', 'inset 0 0 6px rgba(0, 0, 0, 0.3)');
  const borderColor = useColorModeValue('gray.700', 'gray.700');

  const handlePrev = () => {
    if (prevLink) {
      router.push(prevLink);
    }
  };

  const handleNext = () => {
    if (nextLink) {
      router.push(nextLink);
    }
  };

  // Dynamically set sidebar width based on collapsed state
  const sidebarWidth = isCollapsed ? '4rem' : '14rem';

  return (
    <>
      <Head>
        <title>Enterprise AutoML</title>
        <meta name="description" content="Enterprise-Grade AI & Data Science Workspace" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <Flex minH="100vh" bg={layoutBg}>
        {/* Sidebar with dynamic width */}
        <Sidebar isCollapsed={isCollapsed} setIsCollapsed={setIsCollapsed} />

        {/* Main content area shifts based on sidebar width */}
        <Flex direction="column" flex="1" ml={sidebarWidth}>
          <Navbar />

          {/* Skip to content link for improved accessibility */}
          <ChakraLink
            href="#main-content"
            position="absolute"
            top="-40px"
            left="0"
            bg="teal.400"
            color="white"
            p="8px"
            zIndex="100"
            _focus={{ top: '0' }}
          >
            Skip to content
          </ChakraLink>

          {/* Main content container */}
          <Box
            as="main"
            id="main-content"
            role="main"
            flex="1"
            p={{ base: 4, md: 6 }}
            boxShadow={contentBoxShadow}
            borderTopLeftRadius="md"
            mb="80px"
          >
            {children}
          </Box>

          {/* Footer with Prev/Next navigation */}
          <Box
            as="footer"
            py={4}
            px={6}
            bg="gray.800"
            borderTop="1px solid"
            borderColor={borderColor}
            position="fixed"
            bottom="0"
            left={sidebarWidth}
            width={`calc(100% - ${sidebarWidth})`}
            zIndex="1000"
          >
            <HStack justifyContent="space-between">
              {prevLink ? (
                <Button colorScheme="teal" onClick={handlePrev}>
                  Previous
                </Button>
              ) : (
                <Box />
              )}
              {nextLink ? (
                <Button colorScheme="teal" onClick={handleNext}>
                  Next
                </Button>
              ) : (
                <Box />
              )}
            </HStack>
          </Box>
        </Flex>
      </Flex>
    </>
  );
}
