// pages/_app.js
import { ChakraProvider } from '@chakra-ui/react';
import customTheme from '../theme'; // If your theme is /frontend/theme.js
import '../styles/globals.css';     // optional, if you have global CSS

function MyApp({ Component, pageProps }) {
  return (
    <ChakraProvider theme={customTheme}>
      <Component {...pageProps} />
    </ChakraProvider>
  );
}

export default MyApp;
