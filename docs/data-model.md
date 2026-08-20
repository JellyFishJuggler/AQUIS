# AQUIS Data Model

## Tables

### stations
Stable telemetry station metadata from NWDP API.

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PK | Internal ID |
| external_station_id | TEXT UNIQUE | NWDP Station name/key |
| station_name | TEXT | Human-readable name |
| agency | TEXT | Operating agency (e.g. UPGW) |
| state | TEXT | State name |
| state_lgd_code | TEXT | LGD code |
| district | TEXT | District name |
| district_lgd_code | TEXT | LGD code |
| latitude | NUMERIC(12,8) | GPS latitude |
| longitude | NUMERIC(12,8) | GPS longitude |
| first_observed_at | TIMESTAMPTZ | Earliest observation |
| last_observed_at | TIMESTAMPTZ | Latest observation |
| observation_count | INTEGER | Total observations |

### telemetry_observations
Time-series groundwater level readings.

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PK | Internal ID |
| station_id | INTEGER FK | References stations.id |
| observed_at | TIMESTAMPTZ | Observation timestamp |
| groundwater_level | NUMERIC(10,4) | Level in meters |
| source | TEXT | 'nwdp_api' |
| source_record_id | TEXT | NWDP _id field |
| ingestion_run_id | INTEGER FK | References ingestion_runs.id |

**Unique constraint:** (station_id, observed_at)

### assessment_units
Stable geography for assessment data.

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PK | Internal ID |
| external_unit_id | TEXT UNIQUE | Composite key: STATE|DISTRICT|UNIT |
| state | TEXT | State |
| district | TEXT | District |
| assessment_unit | TEXT | Assessment unit name |

### assessment_records
Multi-year assessment data from CentralReport Excel files.

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PK | Internal ID |
| assessment_unit_id | INTEGER FK | References assessment_units.id |
| assessment_year | TEXT | e.g. '2024-2025' |
| rainfall_mm | NUMERIC | Rainfall |
| recharge_*_ham | NUMERIC | Various recharge components |
| total_extraction_ham | NUMERIC | Total GW extraction |
| extraction_rate_pct | NUMERIC | Stage of extraction |
| status | stress_category | SAFE/SEMI/CRITICAL/OVER |
| raw_data | JSONB | Original Excel row |
| source_file | TEXT | Source Excel filename |

**Unique constraint:** (assessment_unit_id, assessment_year)

### ingestion_runs
Tracks every ingestion operation.

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PK | Internal ID |
| source | TEXT | 'telemetry_api' or 'assessment_excel' |
| status | ingestion_status | running/completed/failed |
| records_seen/inserted/rejected/duplicates | INTEGER | Counters |
| error_summary | JSONB | Error details |

### data_quality
Quality issues detected during ingestion or validation.

### model_metadata
ML model registry (task, name, version, metrics, features).

### model_outputs
Stored predictions (forecasts, anomalies, risk assessments).

### groundwater_data
Legacy table preserved for backward compatibility.

## Enums

```sql
stress_category: SAFE, SEMI_CRITICAL, CRITICAL, OVER_EXPLOITED
ingestion_status: running, completed, failed, partial
model_status: training, trained, evaluated, selected, deprecated, failed
```
