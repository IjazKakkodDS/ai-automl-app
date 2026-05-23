# frontend/Dockerfile
#
# Next.js 15 frontend for the AI AutoML Intelligence Platform.
# Local portfolio/demo use only -- not hardened for production.
#
# The frontend makes browser-side API calls to the backend at
# http://127.0.0.1:8000 (hardcoded in frontend/lib/api.js).
# When running via docker-compose, map backend port 8000 to the
# host so the browser can reach it at localhost:8000.

FROM node:20-slim

WORKDIR /app

# Disable Next.js telemetry at build and runtime
ENV NEXT_TELEMETRY_DISABLED=1

# Copy dependency manifests first to leverage Docker layer cache.
# package-lock.json is present; npm ci installs the exact locked versions.
COPY package.json package-lock.json ./

RUN npm ci

# Copy the rest of the frontend source.
# frontend/node_modules and frontend/.next are excluded via .dockerignore.
COPY . .

# Build the Next.js production bundle.
# No API calls are made at build time (all data fetching is client-side).
RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start"]
