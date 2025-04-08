import React from 'react';
import {
  Flex,
  Heading,
  Spacer,
  IconButton,
  useColorMode,
  useColorModeValue,
  Link as ChakraLink,
} from '@chakra-ui/react';
import { SunIcon, MoonIcon } from '@chakra-ui/icons';
import Link from 'next/link';

export default function Navbar() {
  const { colorMode, toggleColorMode } = useColorMode();
  const navBg = useColorModeValue('gray.800', 'gray.800');
  const borderColor = useColorModeValue('gray.700', 'gray.700');
  const textColor = useColorModeValue('teal.300', 'teal.300');

  return (
    <Flex
      as="nav"
      bg={navBg}
      borderBottom="1px solid"
      borderColor={borderColor}
      align="center"
      px={4}
      py={3}
      boxShadow="md"
    >
      {/* Single Link with a single ChakraLink */}
      <Link href="/" legacyBehavior>
        <ChakraLink _hover={{ textDecoration: 'none' }}>
          <Heading as="h2" size="md" color={textColor}>
            AI Workspace
          </Heading>
        </ChakraLink>
      </Link>

      <Spacer />

      <IconButton
        aria-label="Toggle color mode"
        icon={colorMode === 'light' ? <MoonIcon /> : <SunIcon />}
        onClick={toggleColorMode}
        variant="ghost"
        size="lg"
        color="teal.300"
        _hover={{ bg: 'gray.700' }}
      />
    </Flex>
  );
}
