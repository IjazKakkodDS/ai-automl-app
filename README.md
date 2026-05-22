
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![ML Pipeline](https://img.shields.io/badge/ML--Pipeline-scikit--learn%20%7C%20statsmodels%20%7C%20Prophet%20%7C%20FAISS%20%7C%20SentenceTransformers-orange)
![Status](https://img.shields.io/badge/Status-Deployment--Ready-brightgreen)
![Docker](https://img.shields.io/badge/Container-Docker-blue?logo=docker)
![AWS](https://img.shields.io/badge/Cloud-AWS-orange?logo=amazon-aws)

# AI AutoML Intelligence Platform

**FAISS RAG • Agentic AI • Full-Stack ML Platform**
FastAPI + Next.js + Docker | Python 3.12+

---

## 📚 Table of Contents

- [Overview](#overview)
- [Problem Statement &amp; Objectives](#problem-statement--objectives)
- [Key Features &amp; Impact](#key-features--impact)
- [Project Highlights](#project-highlights)
- [Pipeline Breakdown](#pipeline-breakdown)
- [Technical Depth](#technical-depth)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Quick Start](#quick-start)
- [Deployment Guide](#deployment-guide)
- [RAG Scraper: Research Summaries](#rag-scraper-research-summaries)
- [Limitations &amp; Future Work](#limitations--future-work)
- [Author](#author)
- [License](#license)

---

## Overview

This project is a production-grade **AI AutoML Intelligence Platform** that integrates the complete machine learning lifecycle into a unified FastAPI-powered pipeline. It automates **data preprocessing**, **EDA**, **feature engineering**, **model training**, **time-series forecasting**, and **automated reporting**.

It also includes:

- RAG-based **document QA** using FAISS vector search, SentenceTransformer embeddings, and Ollama local LLM summarization
- AI **agent collaboration** for smart automation (EDA, modeling, feature engineering)
- Containerized deployment via **Docker**

**Goal**: Provide a scalable, reproducible, and easily extensible framework for end-to-end machine learning in a single application.

---

## Problem Statement & Objectives

Traditional ML workflows often involve scattered scripts and notebooks, leading to:

- Redundant code and inconsistent data handling
- Difficulties in reproducing results
- Complex deployment pipelines

**This project aims to**:

1. Unify the entire ML lifecycle into a single pipeline.
2. Automate key steps (preprocessing, EDA, modeling, and reporting).
3. Enable scalable, deployment-ready code structure.

---

## Key Features & Impact

- ✅ **Automated Data-to-Report Pipeline**Transform raw data into actionable insights and final stakeholder reports.
- 📊 **AI-Powered EDA**Generate dynamic descriptive stats, correlations, and visuals, with AI-generated narratives for deeper insights.
- 🔄 **Time-Series Forecasting**Tools like Prophet, ARIMA, and LSTM for advanced, modular forecasting.
- ✨ **RAG-Based Knowledge Retrieval**Retrieve ML insights from academic literature via FAISS vector search, SentenceTransformer embeddings, and Ollama local LLM summarization.
- ✉ **Automated Reporting**Produce PDF/HTML reports using headless Chrome or similar solutions.
- 💡 **Extensible Architecture**
  Swap or customize models and modules (e.g., XGBoost, RandomForest) with minimal code changes.

---

## Project Highlights

1. **Unified ML-to-Report Pipeline**From raw data ingestion to final PDF/HTML reports, all in one codebase.
2. **Agentic Intelligence**Orchestrated specialist agents for EDA, feature engineering, model training, and AI insights -- coordinated by an orchestrator decision loop.
3. **Docker Service Packaging**Containerize and build the backend service with a provided Dockerfile for local and cloud deployment.
4. **Scalable Forecasting**Modular time-series forecasting, comparing models like Prophet and ARIMA with minimal setup.
5. **Smart Automation**
   Hyperparameter tuning (Optuna), background tasks for PDF generation, integrated logs, and monitoring.

---

## Pipeline Breakdown

### 1. Data Preprocessing

- Null handling (mean/median or domain-driven)
- Type conversion, categorical encoding
- Scaling (MinMax, Standard)

### 2. Exploratory Data Analysis (EDA)

- Descriptive stats & correlation
- Visual plots: histograms, scatter, boxplots
- Statistical tests + automated report logging

### 3. Feature Engineering

- Date/time parsing
- Polynomial & interaction terms
- Outlier treatment
- Cardinality reduction

### 4. Model Training

- Train/test splits, cross-validation
- Models: Logistic, XGBoost, Random Forest, LightGBM, etc.
- Hyperparameter tuning (Optuna)
- Model saving & versioning for rollback

### 5. Time Series Forecasting

- Resampling + ARIMA/Prophet models
- Holiday/event awareness
- Drift detection support

### 6. Automated Reporting

- Charts, EDA tables, metrics, forecasts
- Export to PDF & HTML (headless Chrome or similar)
- Saved to `/reports/` folder

---

## Technical Depth

- **Reproducibility**Deterministic seeds, thorough logging, version-controlled artifacts.
- **Modular Architecture**Each pipeline step (preprocessing, modeling, reporting) is isolated and easy to replace or extend.
- **Logging and Monitoring**Structured logging throughout all agents and API routes, with timestamped output to file and console.
- **Data Validation**`pydantic` for type enforcement and schema checks.
- **Model Management**Timestamp-based versioning, enabling rollbacks and experiment comparisons.
- **Scalable Forecasting**Designed to handle periodic retraining with optional drift-detection hooks.
- **Asynchronous Reporting**
  CPU-intensive PDF generation tasks run in the background to keep the API responsive.

---

## Tech Stack

- **Backend**: FastAPI (Python 3.12+)
- **Frontend**: Next.js (React), Tailwind CSS
- **ML Libraries**: scikit-learn, statsmodels, Prophet, FAISS, SentenceTransformers
- **Orchestration**: Docker
- **Agentic AI**: FAISS vector search, SentenceTransformer embeddings, Ollama local LLM

---

## Folder Structure

```bash
ai-automl-app/
├── backend/                 # FastAPI code for ML pipeline & endpoints
│   ├── app/                 # Main application
│   ├── models/              # Training scripts, model classes
│   ├── utils/               # Utility functions
│   └── ...
├── frontend/                # Next.js UI with charts, dashboards, etc.
│   ├── components/          # Reusable React components
│   ├── pages/               # Page routes
│   └── ...
├── models/                  # Saved model objects
├── reports/                 # Generated PDF/HTML dashboards
├── processed_data/          # Cleaned or processed datasets
├── scrape_documents.py      # Script for RAG-based doc scraping
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have **Python 3.12+** and **Node.js 14+** (or later) installed.
2. **Run Backend (FastAPI)**

   ```bash
   uvicorn backend.app.main:app --reload
   ```

   Access API docs at [http://localhost:8000/docs](http://localhost:8000/docs).
3. **Run Frontend (Next.js)**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Open [http://localhost:3000](http://localhost:3000) in your browser.

### Environment Variables

If your application requires credentials or special configs, create a `.env` file in `backend` or `frontend` and load them with a library like `python-dotenv` or via Next.js built-in env support.

---

## Deployment Guide

### Docker (Local)

```bash
docker build -t ai-auto-application .
docker run -d -p 8000:8000 ai-auto-application
```

*The container exposes port `8000` by default.*
Update your frontend’s `NEXT_PUBLIC_API_URL` to point to `http://localhost:8000` if they run separately.

### AWS Elastic Beanstalk

1. **Install CLI Tools**
   ```bash
   pip install awsebcli
   aws configure
   ```
2. **Initialize**
   ```bash
   eb init -p docker ai-auto-application --region <your-region>
   ```
3. **Create & Deploy**
   ```bash
   eb create ai-auto-application-env
   eb deploy
   ```

---

## RAG Scraper: Research Summaries

Use `scrape_documents.py` to fetch ML abstracts, deduplicate them, and embed/rank with FAISS vector search:

```bash
python scrape_documents.py --source <source_name> --limit <num_papers>
```

- **Embeddings & Vector DB**: Useful for semantic search and knowledge retrieval.
- **Integration**: Great for brainstorming new feature ideas or generating relevant prompts.

---

## Limitations & Future Work

- **In-Memory Processing**For large datasets, consider chunked or distributed approaches.
- **Model Drift Monitoring**Add real-time or scheduled drift detection to maintain performance over time.
- **Security**No built-in endpoint auth; implement OAuth2 or tokens for production-grade environments.
- **Expanding Agentic AI**Explore advanced ReAct or AutoGPT-style flows for broader autonomy.
- **Further Cloud Integration**
  Deeper integration with AWS S3, Lambda, or other services for data ingestion and orchestration.

---

## Author

**Ijaz Kakkod**
MSc Data Science & Analytics | Actuarial Analyst → Data Scientist
[LinkedIn](https://linkedin.com/in/ijazkakkod) | [Portfolio](https://notion.so/ijazkakkod) | [GitHub](https://github.com/IjazKakkodDS)

---

## License

This project is **not open source**. All rights are reserved by the author.

- Personal and educational usage is allowed.
- Commercial usage, redistribution, or any derivative work requires **explicit permission** from the author.
- Contact via [LinkedIn](https://linkedin.com/in/ijazkakkod) for collaboration or licensing queries.
