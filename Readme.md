# AQUIS — Aquifer Query and User Information System

A role-based groundwater monitoring platform built on the CGWB DWLR 2023 dataset.
Published research: DOI [10.55041/IJSREM62854](https://ijsrem.com) · IJSREM Vol. 10, May 2026

**Authors:** Srijan Anand Gupta, Utkarsh Srivastava — BPIT, Rohini, New Delhi

---

## What it does

AQUIS makes India's CGWB groundwater data accessible to three user groups:

| Role | Primary need | Interface |
|---|---|---|
| Public / Citizen | Local groundwater status | District picker, status badge, charts |
| District Officer | Anomaly monitoring, alerts | Extraction-rate dashboard, hotspots |
| Researcher | Correlation & trend analysis | Analytical workspace, export |

Classification follows CGWB thresholds:

| Status | Extraction Rate |
|---|---|
| Safe | < 70% |
| Semi-Critical | 70–90% |
| Critical | 90–100% |
| Over-Exploited | > 100% |

---

## Project structure

```
aquis-backend/
├── server.js                        ← entry point
├── app.js                           ← express + middleware + routes
├── .env.example                     ← copy to .env
│
├── db/
│   ├── pool.js                      ← pg connection pool
│   └── schema.sql                   ← groundwater_data table, indexes, triggers
│
├── services/
│   ├── groundwaterClassification.js ← SAFE / SEMI_CRITICAL / CRITICAL / OVER_EXPLOITED
│   └── groundwaterRepository.js     ← all SQL: CRUD + aggregations
│
├── controllers/
│   ├── data.controller.js
│   ├── analytics.controller.js
│   └── alert.controller.js
│
├── routes/
│   ├── data.routes.js               ← /data
│   ├── analytics.routes.js          ← /analytics/*
│   └── alert.routes.js              ← /alerts
│
├── middleware/
│   ├── validators.js                ← express-validator chains
│   └── errorHandler.js
│
└── scripts/
    ├── migrate.js                   ← run schema.sql once
    └── importData.js                ← Excel → clean → classify → PostgreSQL

index.html                           ← public user dashboard (standalone)
```

---

## Setup

### 1. Prerequisites
- Node.js 18+
- PostgreSQL 14+
- Database named `aquis` already created

### 2. Install
```bash
cd aquis-backend
npm install
```

### 3. Configure environment
```bash
cp .env.example .env
```
Edit `.env`:
```
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=aquis
PORT=3000
NODE_ENV=development
```

### 4. Create the table
```bash
npm run migrate
```

### 5. Import dataset
```bash
npm run import -- /path/to/ground_water_dataset.xlsx
```
Reads the `Officer_view` sheet, derives extraction rate, classifies each district, bulk-inserts into `groundwater_data`.

### 6. Start
```bash
npm start
```
API runs at `http://localhost:3000`

---

## API reference

### Data
```
GET    /data                          all records (paginated)
GET    /data?state=Delhi              filter by state
GET    /data?district=Arwal           filter by district
GET    /data?status=OVER_EXPLOITED    filter by status
GET    /data?sort_by=extraction_rate_pct&order=desc
GET    /data/:id                      single record
POST   /data                          insert record
PUT    /data/:id                      partial update (status auto-recomputed)
DELETE /data/:id                      delete record
```

### Analytics
```
GET    /analytics/state-summary               avg extraction, status counts per state
GET    /analytics/state-summary?state=Kerala  single state
GET    /analytics/hotspots?limit=10           top N over-exploited districts
GET    /analytics/status-distribution         national SAFE/SEMI/CRITICAL/OVER breakdown
```

### Alerts
```
GET    /alerts?limit=20     CRITICAL + OVER_EXPLOITED districts, sorted by extraction rate
GET    /alerts/summary      count by severity
```

### Health
```
GET    /health
```

### Example curl requests
```bash
# All over-exploited districts, worst first
curl "http://localhost:3000/data?status=OVER_EXPLOITED&sort_by=extraction_rate_pct&order=desc"

# Top 10 hotspots
curl "http://localhost:3000/analytics/hotspots?limit=10"

# National status distribution
curl "http://localhost:3000/analytics/status-distribution"

# Active alerts
curl "http://localhost:3000/alerts"

# Add a record
curl -X POST http://localhost:3000/data \
  -H "Content-Type: application/json" \
  -d '{"state":"Delhi","district":"South Delhi","total_extraction_ham":9200,"annual_extractable_gw_ham":6100}'
```

---

## Frontend

`index.html` is a standalone public user dashboard. No build step — open directly in a browser.

**Features:**
- District selector (loads from API, falls back to 30 seeded CGWB districts)
- Alert banner — green / amber / orange / red based on status
- Hero card with extraction rate %
- Donut chart (used vs available) + bar chart (recharge vs extraction)
- Nearby stations list with status pills and mini bars
- Profile page with Change Location modal
- Persists selected district in `localStorage`

**To connect to live API:** the file already fetches from `http://localhost:3000` on load. Just run the backend and open the file.

---

## Key data facts (CGWB DWLR 2023, n = 730)

- **85.3%** of districts are Safe
- **7.6%** are Over-Exploited
- Rainfall correlation with groundwater availability: r = 0.003 (negligible)
- Highest stress: West Delhi (290%), Arwal Bihar (274%), Kavaratti (273%)
- Stress is driven by extraction intensity, not rainfall

---

## Tech stack

| Layer | Stack |
|---|---|
| Backend | Node.js, Express.js |
| Database | PostgreSQL |
| Validation | express-validator |
| Frontend | Vanilla HTML/CSS/JS, Chart.js |
| Dataset | CGWB DWLR 2023 (Excel) |

---

## What's not built yet

- JWT authentication / login flow
- Live map (Leaflet)
- Voice assistant
- Officer and Researcher dashboard views
- React Native mobile app
- Live DWLR telemetry integration
- Predictive analytics / early-warning

---

## Paper

> Srijan Anand Gupta, Utkarsh Srivastava.
> *AQUIS: A Role-Based Groundwater Monitoring and Information Interface for CGWB DWLR Data.*
> IJSREM Vol. 10 Issue 05, May 2026. DOI: 10.55041/IJSREM62854
