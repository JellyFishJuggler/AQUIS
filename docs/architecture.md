# AQUIS Architecture

## System Overview

AQUIS (Aquifer Query and User Information System) is a role-based groundwater monitoring platform with three layers:

1. **Backend (Node.js + Express)** - API server, data ingestion, statistics, ML gateway
2. **ML Service (Python + Flask)** - Machine learning models for forecasting, anomaly detection, risk assessment
3. **Frontend (Next.js)** - User interface for public, officer, and researcher roles

## Data Flow

```
RAW DATA
  ↓
INGESTION (telemetry API / assessment Excel)
  ↓
CLEANING + VALIDATION
  ↓
POSTGRESQL (canonical tables)
  ↓
STATISTICAL ANALYSIS (Mann-Kendall, Sen's slope)
  ↓
PYTHON ML (forecasting, anomaly detection, risk)
  ↓
MODEL OUTPUTS → PostgreSQL
  ↓
EXPRESS APIs → Frontend
```

## Components

### Backend Services

| Service | Purpose |
|---------|---------|
| `stationService` | Station CRUD, geospatial queries, upserts |
| `telemetryService` | Observation CRUD, time-series queries |
| `assessmentService` | Assessment unit + multi-year record management |
| `telemetryIngestion` | NWDP API paginated ingestion with retry/dedup |
| `assessmentIngestion` | Excel file parsing, multi-year import |
| `statisticsService` | Mann-Kendall test, Sen's slope estimator |
| `dataQualityService` | Quality checks for telemetry and assessment data |
| `ingestionRunService` | Ingestion run tracking and logging |
| `mlGateway` | HTTP client for Python ML service |
| `modelService` | Model metadata and output management |
| `groundwaterRepository` | Legacy groundwater_data table CRUD |
| `groundwaterClassification` | CGWB extraction rate classification |

### Python ML Service

| Module | Purpose |
|--------|---------|
| `forecast.py` | Linear regression + Random Forest forecasting |
| `anomaly.py` | Isolation Forest anomaly detection |
| `risk.py` | Random Forest classifier for deterioration risk |
| `model_comparison.py` | Model comparison utilities |

## API Route Groups

| Prefix | Purpose |
|--------|---------|
| `/stations` | Station listing, details, nearby, summaries |
| `/telemetry` | Time-series observations |
| `/assessments` | Multi-year assessment records |
| `/trends` | Mann-Kendall + Sen's slope analysis |
| `/ml` | Forecast, anomaly, risk, model metadata |
| `/ml-data` | Clean ML-ready datasets |
| `/data-quality` | Data quality reports |
| `/ingestion` | Trigger and monitor ingestion runs |
| `/data` | Legacy groundwater_data endpoints |
| `/analytics` | Legacy analytics (state summary, hotspots) |
| `/alerts` | Legacy alert endpoints |

## Communication Pattern

### Node ↔ Python ML

```
Express → HTTP POST → Flask ML Service
Express ← JSON response ← Flask ML Service
```

The ML service is independently deployable. Node handles:
- Sending clean data
- Receiving results
- Storing model outputs in PostgreSQL
- Serving results to frontend

Python handles:
- Model training
- Model comparison
- Feature engineering
- Prediction
