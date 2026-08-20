# AQUIS ML Contract

## Communication

Node.js backend communicates with the Python ML service via HTTP.

**Default URL:** `http://localhost:5000`
**Configurable via:** `ML_SERVICE_URL` environment variable

## Endpoints

### Health Check
```
GET /health → { status: "ok", service: "aquis-ml", version: "1.0.0" }
```

### Forecast
```
POST /forecast/:stationId
Body: { model_id?: number }

Response:
{
  station_id: number,
  model: "linear_regression" | "random_forest",
  models_compared: { [model_name]: { rmse, mae, r2 } },
  predictions: [{ step, prediction }],
  training_size: number,
  evaluation_size: number
}
```

### Anomaly Detection
```
POST /anomalies/:stationId
Body: { }

Response:
{
  total_observations: number,
  anomaly_count: number,
  anomaly_rate: number,
  baseline_mean: number,
  baseline_std: number,
  anomalies: [{ index, groundwater_level, anomaly_score, is_anomaly }]
}
```

### Risk Assessment
```
POST /risk/:unitId
Body: { }

Response:
{
  unit_id: number,
  current_status: string,
  current_stage_of_extraction: number,
  stage_change: number,
  risk_score: number (0-1),
  risk_category: "low" | "medium" | "high" | "critical",
  years_analyzed: number,
  assessment_years: string[]
}
```

### Training
```
POST /train
Body: { task: "forecast" | "anomaly" | "risk" }

Response:
{
  status: "success" | "insufficient_data",
  task: string,
  model_type: string,
  training_samples: number,
  features: string[],
  feature_importance: { [feature]: number }
}
```

## Error Handling

Node.js handles these ML service states:
- **Service unavailable** → returns `{ status: "unavailable" }` to frontend
- **Timeout** (60s default) → returns error
- **Malformed response** → returns error
- **Model not trained** → returns `{ status: "unavailable" }`

## Data Flow

```
Node.js: Fetches clean data from PostgreSQL
    ↓
Node.js: Sends to Python via HTTP POST
    ↓
Python: Fetches data from Node.js /ml-data endpoints (optional)
Python: Trains model or generates predictions
    ↓
Python: Returns JSON results
    ↓
Node.js: Validates response structure
Node.js: Stores results in model_outputs table
Node.js: Returns to frontend
```
