# Docker Smoke Test Evidence
**Phase:** 6B.3D  
**Date:** 2026-05-23  
**Tool:** Docker 29.3.0, Docker Compose v5.1.0  
**Host OS:** Windows 11 Home 10.0.26200 (Docker Desktop with WSL2)  
**Scope:** Local portfolio packaging only. Not a production or cloud deployment.

---

## Overview

This document records the Docker build and smoke-test results for the AI AutoML
Intelligence Platform. Both the FastAPI backend and the Next.js 15 frontend were
packaged as Linux containers and verified to serve correct responses. Ollama is
NOT containerized; it runs on the host and is reached via
`host.docker.internal:11434`.

---

## 1. Images Built

`docker compose build` was run from the project root. Both images were built
successfully (exit code 0).

```
docker compose build
```

**Build output (final lines, from task log):**

```
#12 [backend 5/5] RUN grep -v "^pywin32" requirements.txt > /tmp/req_linux.txt \
    && pip install -r /tmp/req_linux.txt
#12 DONE 421.2s

#23 [backend] exporting to image
#23 exporting layers 348.0s done
#23 naming to docker.io/library/automl-backend:local done
#23 unpacking to docker.io/library/automl-backend:local 160.0s done
#23 DONE 510.2s

 Image automl-backend:local Built
 Image automl-frontend:local Built
```

**docker image ls (after build):**

```
REPOSITORY       TAG     IMAGE ID       SIZE     CREATED AT
automl-backend   local   230e8c93f4d4   17.6GB   2026-05-23 17:06:40 BST
automl-frontend  local   259ee6090462    3.88GB   2026-05-23 17:03:56 BST
```

**Note on image size:** The backend image includes PyTorch 2.6.0, CUDA runtime
libraries, ChromaDB, sentence-transformers, Prophet, and the full ML stack
(183 packages). The 17.6 GB virtual size reflects these large ML dependencies.
The frontend image includes Node 20, npm packages, and the pre-built Next.js
production bundle.

---

## 2. pywin32 Filter Fix

The project's `requirements.txt` was generated on Windows and includes
`pywin32==310`, which has no Linux wheel. The root `Dockerfile` filters it
before installing on the Linux container:

```dockerfile
RUN grep -v "^pywin32" requirements.txt > /tmp/req_linux.txt \
    && pip install -r /tmp/req_linux.txt
```

`pip install` output confirmed all 183 packages installed successfully.

---

## 3. Backend Container Smoke Test

Container started from `automl-backend:local`, port 8000 inside mapped to 8001
on host (to avoid conflict with any parallel build process). Environment mirrors
what `docker-compose.yml` specifies for the `backend` service.

**Start command:**
```
docker run -d --name automl-backend-ev \
  -p 8001:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=mistral \
  -e CORS_ALLOWED_ORIGINS=http://localhost:3000 \
  -e BACKEND_HOST=0.0.0.0 \
  -e BACKEND_PORT=8000 \
  automl-backend:local
```

**Root endpoint (GET /):**
```
HTTP 200
{"message":"AI AutoML Backend is running. This is the RAG + Agentic AI pipeline."}
```

**Container status:**
```
NAMES               IMAGE                  STATUS         PORTS
automl-backend-ev   automl-backend:local   Up 38 seconds  0.0.0.0:8001->8000/tcp
```

**Backend startup logs (last lines):**
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     172.17.0.1:42962 - "GET / HTTP/1.1" 200 OK
```

Additional log lines show sentence-transformers model resolution against
HuggingFace (expected startup behavior for the RAG pipeline).

---

## 4. Frontend Container Smoke Test

Container started from `automl-frontend:local`, port 3000 inside mapped to 3001
on host.

**Start command:**
```
docker run -d --name automl-frontend-ev \
  -p 3001:3000 \
  -e NEXT_TELEMETRY_DISABLED=1 \
  automl-frontend:local
```

**Root endpoint (GET /):**
```
HTTP 200 - 62,166 bytes (Next.js HTML response)
```

**Container status:**
```
NAMES                IMAGE                   STATUS         PORTS
automl-frontend-ev   automl-frontend:local   Up 26 seconds  0.0.0.0:3001->3000/tcp
```

**Frontend startup logs:**
```
> frontend@1.0.0 start
> next start

   ▲ Next.js 15.2.1
   - Local:        http://localhost:3000
   - Network:      http://172.17.0.3:3000

 ✓ Starting...
 ✓ Ready in 677ms
```

---

## 5. docker-compose.yml Structure

The compose file (`docker-compose.yml` at project root) defines both services:

| Service  | Build context | Image tag              | Host port | Healthcheck           |
|----------|---------------|------------------------|-----------|-----------------------|
| backend  | `.`           | `automl-backend:local` | 8000      | Python urllib.request |
| frontend | `./frontend`  | `automl-frontend:local`| 3000      | (inherits restart)    |

- `frontend` depends on `backend` with `condition: service_healthy`.
- The backend healthcheck polls `http://localhost:8000/` using Python's stdlib
  (no curl required in the slim image).
- Ollama is accessed via `OLLAMA_HOST=http://host.docker.internal:11434`. It is
  NOT containerized. Endpoints that require Ollama (`/ai-insights`, `/rag/query`)
  return errors when Ollama is not running; all other pipeline stages work
  normally.

---

## 6. .dockerignore

The `.dockerignore` at project root excludes large directories from the build
context:

```
venv/
.venv/
__pycache__/
*.py[cod]
*.log
.DS_Store
.env
.git/
frontend/node_modules/
frontend/.next/
```

This prevents the 400+ MB `frontend/node_modules/` and `.git/` history from
being transferred to the Docker daemon on each build.

---

## 7. Frontend API Configuration

`frontend/lib/api.js` hardcodes `baseURL: 'http://127.0.0.1:8000'`. All API
calls are made browser-side, so the frontend container itself does not need to
reach the backend over the Docker network. When running via `docker compose`,
port 8000 is mapped to the host so the browser can reach `localhost:8000`.
No build-time `NEXT_PUBLIC_API_BASE_URL` variable is required.

---

## 8. docker compose up -d (Full Stack)

After the build phase, `docker compose up -d` was run from the project root and
completed with exit code 0. Both services reached running/healthy state.

**docker compose ps output:**

```
NAME              IMAGE                   COMMAND                SERVICE    STATUS                    PORTS
automl-backend    automl-backend:local    "uvicorn backend.app"  backend    Up 46 seconds (healthy)   0.0.0.0:8000->8000/tcp
automl-frontend   automl-frontend:local   "docker-entrypoint.s"  frontend   Up 25 seconds             0.0.0.0:3000->3000/tcp
```

Key observations:
- `automl-backend` reports `(healthy)` because the Python-based healthcheck
  (`urllib.request.urlopen('http://localhost:8000/')`) returned successfully.
- `automl-frontend` started after backend reached healthy status, per the
  `depends_on: condition: service_healthy` constraint.
- Frontend container uses a port 3000 on the host; browser-side API calls reach
  the backend at `http://127.0.0.1:8000`.

**Compose-mode endpoint verification:**

```
GET http://localhost:8000/
HTTP 200 {"message":"AI AutoML Backend is running. This is the RAG + Agentic AI pipeline."}

GET http://localhost:3000/
HTTP 200 - 62,166 bytes (Next.js HTML)
```

---

## 9. Scope Boundaries

| Claim                                      | Status                   |
|--------------------------------------------|--------------------------|
| Backend Docker image builds on Linux       | Confirmed (exit 0)       |
| Frontend Docker image builds on Linux      | Confirmed (exit 0)       |
| Backend container serves correct JSON      | Confirmed (HTTP 200)     |
| Frontend container serves Next.js HTML     | Confirmed (HTTP 200)     |
| docker-compose.yml `up -d` starts both    | Confirmed (exit 0)       |
| Backend healthcheck passes                 | Confirmed (healthy)      |
| Ollama containerized                       | No - host only           |
| Production or cloud deployment             | No - local/portfolio     |
| Kubernetes or remote registry              | No                       |

---

*Evidence collected 2026-05-23 on the local development machine. Images are
local-only (`automl-backend:local`, `automl-frontend:local`). No registry push
was performed.*
