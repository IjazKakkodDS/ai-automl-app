// Sidebar.js

import React from 'react';
import Link from 'next/link';
import {
  Box,
  Icon,
  Tooltip,
  Heading,
  VStack,
  Text,
  useColorModeValue,
  Button,
  Flex,
  useToast,
} from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { AtSignIcon, InfoIcon } from '@chakra-ui/icons';
import {
  FiMenu,
  FiDatabase,
  FiActivity,
  FiCode,
  FiZap,
  FiTrendingUp,
  FiStar,
  FiFileText,
  FiInfo,
  FiTrash2,
} from 'react-icons/fi';
import { IoArrowBack } from 'react-icons/io5';
import api from '../lib/api';

const MotionBox = motion(Box);

export default function Sidebar({ isCollapsed, setIsCollapsed }) {
  const [resetLoading, setResetLoading] = React.useState(false);
  const toast = useToast();

  // Colors
  const bg = useColorModeValue('gray.800', 'gray.800');
  const borderColor = useColorModeValue('gray.700', 'gray.700');
  const textColor = useColorModeValue('gray.100', 'gray.100');
  const hoverBg = useColorModeValue('gray.700', 'gray.600');
  const accentColor = 'teal.300';

  const links = [
    { label: 'Home', href: '/', icon: AtSignIcon },
    { label: 'Preprocessing', href: '/preprocessing', icon: FiDatabase },
    { label: 'EDA', href: '/eda', icon: FiActivity },
    { label: 'Feature Eng', href: '/feature-engineering', icon: FiCode },
    { label: 'Model Training', href: '/model-training', icon: FiZap },
    { label: 'Forecasting', href: '/forecasting', icon: FiTrendingUp },
    { label: 'Evaluate', href: '/evaluate', icon: FiStar },
    { label: 'AI Insights', href: '/insights', icon: FiInfo },
    { label: 'RAG & Agentic AI', href: '/rag', icon: InfoIcon },
    { label: 'Reports', href: '/reports', icon: FiFileText },
    { label: 'Restart Analysis & Clear Models', action: 'reset', icon: FiTrash2 },
  ];

  // Instead of local state, use the setter from Layout
  const toggleSidebar = () => {
    setIsCollapsed((prev) => !prev);
  };

  const sidebarWidth = isCollapsed ? '4rem' : '14rem';

  const handleReset = async () => {
    if (resetLoading) return;
    setResetLoading(true);
    try {
      const res = await api.post('/reset/restart-analysis');
      const newSessionId = res.data.session_id;
      localStorage.setItem('session_id', newSessionId);
      toast({
        title: 'Analysis restarted',
        description: 'Previous models cleared. New session started.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (err) {
      toast({
        title: 'Reset Failed',
        description: err.response?.data?.detail || 'An error occurred.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setResetLoading(false);
    }
  };

  return (
    <MotionBox
      as="aside"
      position="fixed"
      top="0"
      left="0"
      height="100vh"
      bg={bg}
      color={textColor}
      borderRight="1px solid"
      borderColor={borderColor}
      display="flex"
      flexDirection="column"
      // Animate the width based on isCollapsed
      animate={{ width: sidebarWidth }}
      transition={{ duration: 0.2 }}
      overflow="hidden"
      boxShadow="xl"
      zIndex={10}
    >
      {/* Sidebar Header */}
      <Flex
        align="center"
        justify={isCollapsed ? 'center' : 'space-between'}
        p={3}
        borderBottom="1px solid"
        borderColor={borderColor}
      >
        {!isCollapsed && (
          <Heading as="h3" size="sm" ml={2} color={accentColor}>
            AI Workspace
          </Heading>
        )}
        <Button
          onClick={toggleSidebar}
          variant="ghost"
          size="sm"
          color={accentColor}
          _hover={{ bg: hoverBg }}
        >
          {isCollapsed ? <FiMenu /> : <IoArrowBack />}
        </Button>
      </Flex>

      {/* Sidebar Items */}
      <VStack align="stretch" spacing={1} mt={2}>
        {links.map(({ label, href, icon, action }) => {
          const isResetItem = action === 'reset';
          if (isResetItem) {
            return (
              <Box
                key={label}
                display="flex"
                alignItems="center"
                p={2}
                borderRadius="md"
                ml={2}
                mr={2}
                _hover={{ bg: hoverBg }}
                cursor="pointer"
                onClick={handleReset}
              >
                <Tooltip label={label} placement="right" isDisabled={!isCollapsed} hasArrow>
                  <Icon as={icon} boxSize={5} mr={isCollapsed ? 0 : 3} color={accentColor} />
                </Tooltip>
                {!isCollapsed && (
                  <Text>
                    {resetLoading ? 'Restarting...' : label}
                  </Text>
                )}
              </Box>
            );
          } else {
            return (
              <Link href={href} key={label} passHref>
                <Box
                  display="flex"
                  alignItems="center"
                  p={2}
                  borderRadius="md"
                  ml={2}
                  mr={2}
                  _hover={{ bg: hoverBg }}
                  cursor="pointer"
                >
                  <Tooltip label={label} placement="right" isDisabled={!isCollapsed} hasArrow>
                    <Icon as={icon} boxSize={5} mr={isCollapsed ? 0 : 3} color={accentColor} />
                  </Tooltip>
                  {!isCollapsed && <Text>{label}</Text>}
                </Box>
              </Link>
            );
          }
        })}
      </VStack>
    </MotionBox>
  );
}
