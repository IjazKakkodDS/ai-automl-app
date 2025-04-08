// theme.js
import { extendTheme } from '@chakra-ui/react';

const customTheme = extendTheme({
  colors: {
    brand: {
      50: '#ddfef6',
      100: '#aef1e3',
      200: '#B2F5EA',
      300: '#81E6D9',
      400: '#26be9e',
      500: '#15a487',
      600: '#118270',
      700: '#0c6055',
      800: '#073f3a',
      900: '#031f1f'
    },
  },
  styles: {
    global: {
      'html, body': {
        bg: 'gray.900',
        color: 'gray.100',
      },
    },
  },
});

export default customTheme;
