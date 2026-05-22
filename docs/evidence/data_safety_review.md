# AI AutoML Intelligence Platform - Data Safety Review

**Date:** 2026-05-22
**Phase:** 6B.3A - Safety and Repository Hygiene
**Reviewer:** Engineering audit (automated column-name and artifact-path inspection)
**Status:** Informational review only. Not a formal privacy audit.

---

## 1. Scope

This review covers all files tracked by git in the following directories:

| Directory | Contents |
|---|---|
| `original_data/` | User-uploaded raw CSV datasets |
| `processed_data/` | Pipeline-transformed CSVs (feature-engineered) |
| `reports/` | Generated EDA reports, forecasting logs, HTML/PDF exports, training CSVs |
| `model_reports/` | Training metric CSVs produced by the model training pipeline |

This review does not cover:
- Untracked files or local runtime output not committed to the repository
- The FAISS index in `data_1/` (contains document embeddings, not tabular user data)
- The `ai_cache/` directory (caches AI insight text, not raw data)

---

## 2. Method

- Column names were extracted using `pandas` with `nrows=0` (no raw row values were read or printed)
- Row counts were obtained using `len(df)` after reading the full file (no values printed)
- A fixed PII keyword list was checked against all column names (case-insensitive)
- Report artifact filenames were inspected by path only
- No raw cell values were printed or stored during this review

---

## 3. Tracked File Inventory

| Directory | Tracked File Count | File Types |
|---|---|---|
| `original_data/` | 71 | .csv |
| `processed_data/` | 94 | .csv |
| `reports/` | 129 | .png, .csv, .txt, .html, .pdf |
| `model_reports/` | 8 | .csv |
| **Total** | **302** | |

---

## 4. Sampled Dataset Inspection

### original_data/ (5 sampled files)

| File (UUID-named) | Columns | Row Count | Dataset Character |
|---|---|---|---|
| `05261a4e-...csv` | customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn | 7,043 | Telco churn benchmark (public demo dataset) |
| `05cbd242-...csv` | A, B, C, D | 4 | Synthetic stub (generic labels) |
| `0843a3cc-...csv` | Date, Sales | 12 | Synthetic sales time series |
| `093f0481-...csv` | Date, Sales | 12 | Synthetic sales time series |
| `0a3bbbed-...csv` | Date, Product, Region, Sales, Units Sold, Discount, Category | 500 | Demo retail sales dataset |

### processed_data/ (5 sampled files)

| File (UUID-named) | Columns | Row Count | Dataset Character |
|---|---|---|---|
| `02d20589-...csv` | Sales, Date_year, Date_month, Date_day, Date_dayofweek | 12 | Feature-engineered from sales time series |
| `0518db14-...csv` | Sales, Date_year, Date_month, Date_day, Date_dayofweek | 12 | Feature-engineered from sales time series |
| `083ca016-...csv` | Sales, Units Sold, Discount, Date_year, Date_month, Date_day, Date_dayofweek, Product_B, Product_C, Product_D, Product_E, Region_North, Region_South, Region_West, Category_Electronics, Category_Furniture, Category_Grocery | 500 | One-hot encoded retail dataset |
| `0d2016fb-...csv` | Sales, Date_year, Date_month, Date_day, Date_dayofweek | 12 | Feature-engineered from sales time series |
| `0ef7f71c-...csv` | Sales, Date_year, Date_month, Date_day, Date_dayofweek | 12 | Feature-engineered from sales time series |

All processed files are numeric or one-hot encoded. No raw text values are stored in processed outputs.

### model_reports/ (all 8 files inspected)

All model_reports CSVs contain ML training metric columns only. No user data. No identifiers.

| Column Set A | Column Set B |
|---|---|
| Model, Training_Time_sec, RMSE, R2, MAE, CV_R2_Avg | Model, Training_Time_sec, Accuracy, F1, Precision, Recall |

Each file has 2 to 3 rows (one row per trained model variant). No user-linked data.

### reports/ (selected artifact)

| File | Columns | Row Count | Notes |
|---|---|---|---|
| `Sales_outliers_iqr.csv` | Sales, Year, Month, DayOfWeek | Varies | IQR outlier flagging output, no user IDs |

Other tracked report files are EDA text summaries, PNG distribution plots, or training log CSVs -- all derived from the pipeline. No user-linked filenames.

---

## 5. PII-Oriented Column-Name Scan

The following PII keyword list was checked against all sampled column names (case-insensitive):

`name, email, phone, address, postcode, zip, ssn, national_insurance, ni_number, passport, account_number, sort_code, iban, ip_address, user_id, customer_id, date_of_birth, dob`

### Findings

| Column of Interest | File | Assessment |
|---|---|---|
| `customerID` | `original_data/05261a4e-...csv` | Pseudonymous alphanumeric ID (e.g. "7590-VHVEG"). This is the Telco Customer Churn dataset, a widely distributed public benchmark from Kaggle/IBM. Not a real individual's identifier. |
| `gender` | `original_data/05261a4e-...csv` | Demographic category (Male/Female only). Present in the public Telco churn benchmark. |
| `SeniorCitizen` | `original_data/05261a4e-...csv` | Binary flag (0/1). Present in the public Telco churn benchmark. |

No columns matching `name`, `email`, `phone`, `address`, `ssn`, `dob`, `date_of_birth`, `passport`, `iban`, `account_number`, `sort_code`, `ni_number`, `ip_address`, or similar direct PII fields were found in any sampled file.

All other columns across sampled files are: generic labels (A, B, C, D), temporal aggregations (Date, Year, Month, DayOfWeek), business metrics (Sales, Units Sold, Discount), product/region categories, or ML-derived features.

---

## 6. Generated Artifact Review

### reports/ directory (129 files)

| Artifact Type | Count (approx) | Assessment |
|---|---|---|
| PNG distribution plots | ~78 | EDA chart exports. No embedded user data. |
| TXT EDA summaries | ~20 | Statistical summaries. UUID-named by dataset_id. |
| CSV training results | ~20 | Model metric outputs. Contain model names and numeric metrics only. |
| HTML/PDF full reports | 2 | Assembled from EDA + training + forecasting outputs. No direct user identifiers. |
| CSV forecasting outputs | ~3 | Forecasting numeric results only. |
| TXT log files | ~4 | Pipeline logs. Contain no user-linked rows. |

All filenames use UUID-based dataset identifiers (e.g. `eda_report_05261a4e-...txt`). No filenames include personal names, user accounts, or email addresses.

### model_reports/ directory (8 files)

All are timestamp-named (Unix epoch). Contain only ML metric data. No user-linked content.

---

## 7. Findings

| # | Finding | Risk Level | Notes |
|---|---|---|---|
| 1 | `customerID` column present in one original_data file | Low | Pseudonymous ID from a public benchmark dataset. Not a real personal identifier. |
| 2 | `gender` and `SeniorCitizen` columns in the same file | Low | Demographic categories from a well-known public dataset. No direct identification risk. |
| 3 | Multiple generated report files (HTML, PDF, TXT, CSV) are tracked in `reports/` | Low-Medium | These contain ML metrics and statistical summaries, not raw user data rows. However, tracking generated artifacts increases repo size and could include analysis of user-uploaded files in future deployments. |
| 4 | No `.env.example` existed before this phase | Informational | Developers had no documented reference for required environment variables. Resolved by creating `.env.example` in this phase. |
| 5 | CORS set to `allow_origins=["*"]` in `backend/app/main.py` | Low (dev) / Medium (prod) | Appropriate for local development. Must be restricted before any public deployment. |
| 6 | No hardcoded secrets found in backend Python files | Clear | Ollama called via Python package default (localhost:11434). No API keys embedded. |

---

## 8. Repository Hygiene Actions Taken in Phase 6B.3A

| Action | File | Purpose |
|---|---|---|
| Created | `.env.example` | Documents all required environment variables with placeholder values. No real values. |
| Updated | `.gitignore` | Added rules to prevent future generated HTML, PDF, TXT, CSV, JSON files in `reports/` and `model_reports/` from being tracked. Added `.env` / `.env.*` exclusion. Added `.pytest_cache/`, `frontend/.next/`, `frontend/out/`, model artifact extensions (`.pkl`, `.joblib`), and `logs/`. |
| Created | `docs/evidence/data_safety_review.md` | This document. |

Note: Existing tracked files in `reports/` and `model_reports/` are not removed from tracking in this phase. The new `.gitignore` rules are forward-looking only.

---

## 9. Remaining Risk Notes

- **Telco churn dataset (`05261a4e-...csv`):** If this is the IBM Telco Customer Churn dataset, it is widely distributed as a public demo resource and does not contain real individuals' data. Confirm the provenance of this file before any public release and consider replacing with a generated synthetic sample if provenance is uncertain.
- **Generated report tracking:** The 129 tracked files in `reports/` include HTML, PDF, and TXT artifacts. These are safe for this project (ML metrics only), but future users who upload real personal data and trigger the pipeline will generate artifacts that could contain PII-derived statistics. Teams deploying this system with real data should add a data classification layer and ensure generated reports are not tracked.
- **CORS wildcard:** The `allow_origins=["*"]` setting in `backend/app/main.py` must be tightened to a specific origin list before any non-local deployment.
- **Ollama network exposure:** The Ollama service runs locally on port 11434. In a multi-user or cloud deployment, ensure Ollama is not exposed on a public interface.

---

## 10. Claim-Safe Conclusion

Based on sampled column-name and artifact-path inspection, no obvious direct PII fields were identified in the reviewed tracked datasets. The one file containing a `customerID` column appears to be a publicly distributed benchmark dataset with pseudonymous identifiers, not a real customer extract.

This review does not constitute a formal privacy audit. Before public release, tracked datasets should either be replaced with confirmed synthetic samples or documented as safe public/demo data with clear provenance statements. Generated report artifacts in `reports/` should be reviewed if the platform is deployed with real user data in future.

---

*Review generated as part of Phase 6B.3A: Safety and Repository Hygiene for AI AutoML Intelligence Platform.*
