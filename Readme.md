# AQUIS — Aquifer Query and User Information System

A role-based groundwater monitoring platform built on NWDP telemetry and CGWB assessment data.

**Authors:** Srijan Anand Gupta, Utkarsh Srivastava — BPIT, Rohini, New Delhi

---

## Architecture

```
back-end/          Node.js + Express API server
ml/                Python ML service (Flask)
front-end/         Next.js 16 frontend
docs/              Documentation
```

**Data flow:**
NWDP Telemetry API + Assessment Excel files → PostgreSQL → Statistical Analysis + ML → Express APIs → Frontend

---

## Setup

### 1. Prerequisites
- Node.js 18+
- PostgreSQL 14+
- Python 3.9+

### 2. Backend
```bash
cd back-end
cp .env.example .env    # Edit with your database credentials
npm install
npm run migrate         # Apply schema
npm run import-assessments   # Import assessment Excel files
npm run import-telemetry     # Import NWDP telemetry data (large dataset)
npm start              # Start API at http://localhost:3000
```

### 3. ML Service
```bash
cd ml
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app.py              # Start ML service at http://localhost:5000
```

### 4. Frontend
```bash
cd front-end
npm install
npm run dev            # Start Next.js at http://localhost:3000
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | — | PostgreSQL connection string |
| PORT | 3000 | Backend port |
| NODE_ENV | development | Environment |
| ML_SERVICE_URL | http://localhost:5000 | Python ML service URL |
| ML_TIMEOUT_MS | 60000 | ML request timeout |

---

## API Reference

### Health
```
GET /health
```

### Stations
```
GET    /stations                          List all stations (paginated)
GET    /stations/:stationId               Station details
GET    /stations/nearby?lat=&lon=         Nearby stations
GET    /stations/state-summary            Station count by state
GET    /stations/district-summary         Station count by district
```

### Telemetry
```
GET    /telemetry                         All observations (filtered)
GET    /telemetry/latest?stationId=       Latest observation for station
GET    /telemetry/summary                 State/district summary
GET    /telemetry/:stationId              History for station
```

### Assessments
```
GET    /assessments                       Assessment records (filtered)
GET    /assessments/:id                   Single record
GET    /assessments/:unitId/history       Multi-year history for unit
GET    /assessments/summary               State/year summary
GET    /assessments/years                 Available assessment years
```

### Trends (Statistical Analysis)
```
GET    /trends/:stationId                 Mann-Kendall + Sen's slope for station
GET    /trends/summary                    Trend summary across all stations
```

### ML - Forecast, Anomaly, Risk
```
GET    /ml/health                         ML service health
GET    /ml/forecast/:stationId            Get forecast
GET    /ml/anomalies                      All anomalies
GET    /ml/anomalies/:stationId           Station anomalies
GET    /ml/anomalies/summary              Anomaly summary
GET    /ml/risk/:unitId                   Risk assessment
GET    /ml/risk/summary                   Risk summary
GET    /ml/risk/priority-areas            High-risk areas
GET    /ml/models                         List models
GET    /ml/models/compare?task=           Model comparison
GET    /ml/models/:name                   Model details
GET    /ml/models/:name/metrics           Model metrics
```

### ML-Ready Data
```
GET    /ml-data/telemetry                 Clean telemetry dataset
GET    /ml-data/telemetry/:stationId      Station-specific telemetry
GET    /ml-data/assessment                Clean assessment dataset
GET    /ml-data/assessment/:unitId        Unit-specific assessment history
```

### Data Quality
```
GET    /data-quality/:stationId           Station quality issues
GET    /data-quality/telemetry            Telemetry quality summary
GET    /data-quality/assessment           Assessment quality summary
```

### Ingestion
```
GET    /ingestion                         List ingestion runs
GET    /ingestion/:id                     Run details
POST   /ingestion/telemetry               Trigger telemetry ingestion
POST   /ingestion/assessment              Trigger assessment ingestion
```

### Legacy
```
GET    /data                              Legacy groundwater_data records
GET    /analytics/state-summary           State analytics
GET    /analytics/hotspots                Top over-exploited districts
GET    /analytics/status-distribution     National status breakdown
GET    /alerts                            Critical + over-exploited alerts
GET    /alerts/summary                    Alert counts by severity
```

---

## Testing
```bash
cd back-end
npm test
```
42 tests covering classification, statistics, telemetry utilities, ML gateway, and app configuration.

---

## Key Data Facts

- NWDP telemetry dataset: ~7.4 million records across India
- Assessment years available: 2016-2017 through 2025-2026
- 154-column CentralReport Excel format with 3-level merged headers
- CGWB classification: Safe (<70%), Semi-Critical (70-90%), Critical (90-100%), Over-Exploited (>100%)

---

## Tech Stack

| Layer | Stack |
|-------|-------|
| Backend | Node.js, Express.js, PostgreSQL |
| ML Service | Python, Flask, scikit-learn, XGBoost |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS v4 |
| Charts | Chart.js |
| Data Sources | NWDP Telemetry API, CGWB Assessment Excel files |
