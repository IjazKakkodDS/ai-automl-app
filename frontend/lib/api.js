// lib/api.js
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000', // your FastAPI backend
});

export default api;
