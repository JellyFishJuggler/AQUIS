-- AQUIS Database Schema v2
-- Comprehensive schema for telemetry, assessments, ML, and analytics.
-- Run: node scripts/migrate.js

-- ═══════════════════════════════════════════════════════════
-- ENUMS
-- ═══════════════════════════════════════════════════════════

DO $$ BEGIN
  CREATE TYPE stress_category AS ENUM (
    'SAFE', 'SEMI_CRITICAL', 'CRITICAL', 'OVER_EXPLOITED'
  );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE ingestion_status AS ENUM (
    'running', 'completed', 'failed', 'partial'
  );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE model_status AS ENUM (
    'training', 'trained', 'evaluated', 'selected', 'deprecated', 'failed'
  );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ═══════════════════════════════════════════════════════════
-- INGESTION RUNS
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS ingestion_runs (
  id              SERIAL PRIMARY KEY,
  source          TEXT NOT NULL,                -- 'telemetry_api', 'assessment_excel'
  source_ref      TEXT,                         -- file path or API resource_id
  status          ingestion_status NOT NULL DEFAULT 'running',
  started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  finished_at     TIMESTAMPTZ,
  records_seen    INTEGER DEFAULT 0,
  records_inserted INTEGER DEFAULT 0,
  records_updated INTEGER DEFAULT 0,
  records_rejected INTEGER DEFAULT 0,
  records_duplicates INTEGER DEFAULT 0,
  error_count     INTEGER DEFAULT 0,
  error_summary   JSONB,
  metadata        JSONB
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_source ON ingestion_runs (source);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_status ON ingestion_runs (status);

-- ═══════════════════════════════════════════════════════════
-- STATIONS (telemetry metadata)
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS stations (
  id                  SERIAL PRIMARY KEY,
  external_station_id TEXT UNIQUE NOT NULL,
  station_name        TEXT,
  agency              TEXT,
  state               TEXT,
  state_lgd_code      TEXT,
  district            TEXT,
  district_lgd_code   TEXT,
  tehsil              TEXT,
  block               TEXT,
  village             TEXT,
  river               TEXT,
  basin               TEXT,
  tributary           TEXT,
  subtributary        TEXT,
  sub_subtributary    TEXT,
  local_river         TEXT,
  latitude            NUMERIC(12, 8),
  longitude           NUMERIC(12, 8),
  rl_msl              NUMERIC(10, 4),
  first_observed_at   TIMESTAMPTZ,
  last_observed_at    TIMESTAMPTZ,
  observation_count   INTEGER DEFAULT 0,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stations_state ON stations (state);
CREATE INDEX IF NOT EXISTS idx_stations_district ON stations (district);
CREATE INDEX IF NOT EXISTS idx_stations_agency ON stations (agency);
CREATE INDEX IF NOT EXISTS idx_stations_coords ON stations (latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_stations_external ON stations (external_station_id);

-- ═══════════════════════════════════════════════════════════
-- TELEMETRY OBSERVATIONS
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS telemetry_observations (
  id                  SERIAL PRIMARY KEY,
  station_id          INTEGER NOT NULL REFERENCES stations(id) ON DELETE CASCADE,
  observed_at         TIMESTAMPTZ NOT NULL,
  groundwater_level   NUMERIC(10, 4),
  source              TEXT DEFAULT 'nwdp_api',
  source_record_id    TEXT,
  ingestion_run_id    INTEGER REFERENCES ingestion_runs(id),
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_telemetry_station_time
  ON telemetry_observations (station_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_observed_at ON telemetry_observations (observed_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_station_id ON telemetry_observations (station_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_source_record ON telemetry_observations (source_record_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_run ON telemetry_observations (ingestion_run_id);

-- ═══════════════════════════════════════════════════════════
-- ASSESSMENT UNITS (stable geography/identity)
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS assessment_units (
  id                  SERIAL PRIMARY KEY,
  external_unit_id    TEXT UNIQUE NOT NULL,      -- state+district+assessment_unit composite key
  state               TEXT NOT NULL,
  district            TEXT NOT NULL,
  assessment_unit     TEXT,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_assessment_units_state ON assessment_units (state);
CREATE INDEX IF NOT EXISTS idx_assessment_units_district ON assessment_units (district);

-- ═══════════════════════════════════════════════════════════
-- ASSESSMENT RECORDS (multi-year)
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS assessment_records (
  id                          SERIAL PRIMARY KEY,
  assessment_unit_id          INTEGER NOT NULL REFERENCES assessment_units(id) ON DELETE CASCADE,
  assessment_year             TEXT NOT NULL,        -- e.g. '2024-2025'
  rainfall_mm                 NUMERIC(12, 4),
  recharge_worthy_area_ha     NUMERIC(14, 4),
  recharge_rainfall_ham       NUMERIC(14, 4),
  recharge_canals_ham         NUMERIC(14, 4),
  recharge_surface_irr_ham    NUMERIC(14, 4),
  recharge_gw_irr_ham         NUMERIC(14, 4),
  recharge_tanks_ponds_ham    NUMERIC(14, 4),
  recharge_wcs_ham            NUMERIC(14, 4),
  recharge_pipelines_ham      NUMERIC(14, 4),
  recharge_sewage_ff_ham      NUMERIC(14, 4),
  total_recharge_ham          NUMERIC(14, 4),
  stream_recharge_ham         NUMERIC(14, 4),
  annual_recharge_ham         NUMERIC(14, 4),
  environmental_flows_ham     NUMERIC(14, 4),
  annual_extractable_gw_ham   NUMERIC(14, 4),
  extraction_domestic_ham     NUMERIC(14, 4),
  extraction_industrial_ham   NUMERIC(14, 4),
  extraction_irrigation_ham   NUMERIC(14, 4),
  total_extraction_ham        NUMERIC(14, 4),
  stage_of_extraction_pct     NUMERIC(8, 4),
  extraction_rate_pct         NUMERIC(8, 4),
  net_availability_future_ham NUMERIC(14, 4),
  allocation_domestic_ham     NUMERIC(14, 4),
  quality_tagging_major       TEXT,
  quality_tagging_other       TEXT,
  coastal_area_ham            NUMERIC(14, 4),
  storage_unconfined_fresh    NUMERIC(14, 4),
  storage_unconfined_saline   NUMERIC(14, 4),
  availability_unconfined_fresh  NUMERIC(14, 4),
  availability_unconfined_saline NUMERIC(14, 4),
  dynamic_confined_fresh      NUMERIC(14, 4),
  dynamic_confined_saline     NUMERIC(14, 4),
  storage_confined_fresh      NUMERIC(14, 4),
  storage_confined_saline     NUMERIC(14, 4),
  total_confined_fresh        NUMERIC(14, 4),
  total_confined_saline       NUMERIC(14, 4),
  dynamic_semiconfined_fresh  NUMERIC(14, 4),
  dynamic_semiconfined_saline NUMERIC(14, 4),
  storage_semiconfined_fresh  NUMERIC(14, 4),
  storage_semiconfined_saline NUMERIC(14, 4),
  total_semiconfined_fresh    NUMERIC(14, 4),
  total_semiconfined_saline   NUMERIC(14, 4),
  total_gw_availability_fresh    NUMERIC(14, 4),
  total_gw_availability_saline   NUMERIC(14, 4),
  total_gw_availability_ham      NUMERIC(14, 4),
  total_geographical_area_ha     NUMERIC(14, 4),
  hilly_area_ha                  NUMERIC(14, 4),
  status                      stress_category,
  raw_data                    JSONB,                -- preserve original Excel row for audit
  source                      TEXT DEFAULT 'centralreport_ingres',
  source_file                 TEXT,
  source_record_id            TEXT,
  ingestion_run_id            INTEGER REFERENCES ingestion_runs(id),
  created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (assessment_unit_id, assessment_year)
);

CREATE INDEX IF NOT EXISTS idx_assessment_records_year ON assessment_records (assessment_year);
CREATE INDEX IF NOT EXISTS idx_assessment_records_unit ON assessment_records (assessment_unit_id);
CREATE INDEX IF NOT EXISTS idx_assessment_records_status ON assessment_records (status);
CREATE INDEX IF NOT EXISTS idx_assessment_records_state ON assessment_records (ingestion_run_id);

-- ═══════════════════════════════════════════════════════════
-- DATA QUALITY
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS data_quality (
  id                  SERIAL PRIMARY KEY,
  entity_type         TEXT NOT NULL,             -- 'telemetry', 'assessment'
  entity_id           INTEGER NOT NULL,
  check_name          TEXT NOT NULL,             -- 'missing_timestamp', 'invalid_coordinates', etc.
  severity            TEXT DEFAULT 'warning',    -- 'info', 'warning', 'error'
  message             TEXT,
  details             JSONB,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dq_entity ON data_quality (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_dq_check ON data_quality (check_name);

-- ═══════════════════════════════════════════════════════════
-- MODEL METADATA
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS model_metadata (
  id                  SERIAL PRIMARY KEY,
  task                TEXT NOT NULL,             -- 'forecast', 'anomaly', 'risk'
  model_name          TEXT NOT NULL,
  model_type          TEXT NOT NULL,             -- 'linear_regression', 'random_forest', 'xgboost', etc.
  version             TEXT NOT NULL,
  training_start      DATE,
  training_end        DATE,
  evaluation_start    DATE,
  evaluation_end      DATE,
  features            JSONB,                     -- array of feature names
  feature_importance  JSONB,                     -- feature -> importance mapping
  metrics             JSONB,                     -- { rmse, mae, r2, etc. }
  is_selected         BOOLEAN DEFAULT FALSE,
  status              model_status DEFAULT 'trained',
  artifact_path       TEXT,
  uncertainty_config  JSONB,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_task ON model_metadata (task);
CREATE INDEX IF NOT EXISTS idx_model_name ON model_metadata (model_name);

-- ═══════════════════════════════════════════════════════════
-- MODEL OUTPUTS
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS model_outputs (
  id                  SERIAL PRIMARY KEY,
  model_id            INTEGER NOT NULL REFERENCES model_metadata(id) ON DELETE CASCADE,
  output_type         TEXT NOT NULL,             -- 'forecast', 'anomaly', 'risk'
  entity_type         TEXT NOT NULL,             -- 'station', 'assessment_unit'
  entity_id           INTEGER NOT NULL,
  predicted_at        TIMESTAMPTZ NOT NULL,      -- when prediction was made
  target_time         TIMESTAMPTZ,               -- for forecasts: what time is being predicted
  target_year         TEXT,                      -- for risk/assessment predictions
  prediction          NUMERIC(14, 4),
  lower_bound         NUMERIC(14, 4),
  upper_bound         NUMERIC(14, 4),
  probability         NUMERIC(8, 6),
  confidence          NUMERIC(8, 6),
  is_anomaly          BOOLEAN,
  anomaly_score       NUMERIC(10, 6),
  risk_category       TEXT,
  details             JSONB,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mo_model ON model_outputs (model_id);
CREATE INDEX IF NOT EXISTS idx_mo_type_entity ON model_outputs (output_type, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_mo_target_time ON model_outputs (target_time);
CREATE INDEX IF NOT EXISTS idx_mo_target_year ON model_outputs (target_year);

-- ═══════════════════════════════════════════════════════════
-- LEGACY TABLE (preserved for backward compatibility)
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS groundwater_data (
  id                          SERIAL PRIMARY KEY,
  state                       TEXT NOT NULL,
  district                    TEXT NOT NULL,
  assessment_unit             TEXT,
  rainfall_mm                 NUMERIC(10, 2),
  recharge_worthy_area_ha     NUMERIC(14, 4),
  recharge_rainfall_ham       NUMERIC(14, 4),
  recharge_canals_ham         NUMERIC(14, 4),
  recharge_surface_irr_ham    NUMERIC(14, 4),
  recharge_gw_irr_ham         NUMERIC(14, 4),
  recharge_tanks_ponds_ham    NUMERIC(14, 4),
  recharge_wcs_ham            NUMERIC(14, 4),
  recharge_pipelines_ham      NUMERIC(14, 4),
  recharge_sewage_ff_ham      NUMERIC(14, 4),
  total_recharge_ham          NUMERIC(14, 4),
  stream_recharge_ham         NUMERIC(14, 4),
  annual_recharge_ham         NUMERIC(14, 4),
  environmental_flows_ham     NUMERIC(14, 4),
  annual_extractable_gw_ham   NUMERIC(14, 4),
  extraction_domestic_ham     NUMERIC(14, 4),
  extraction_industrial_ham   NUMERIC(14, 4),
  extraction_irrigation_ham   NUMERIC(14, 4),
  total_extraction_ham        NUMERIC(14, 4),
  extraction_rate_pct         NUMERIC(8, 4),
  net_availability_future_ham NUMERIC(14, 4),
  storage_unconfined_fresh_ham   NUMERIC(14, 4),
  availability_unconfined_fresh  NUMERIC(14, 4),
  availability_confined_fresh    NUMERIC(14, 4),
  availability_semiconfined_fresh NUMERIC(14, 4),
  total_gw_availability_ham      NUMERIC(14, 4),
  status                      stress_category,
  created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gw_state       ON groundwater_data (state);
CREATE INDEX IF NOT EXISTS idx_gw_district    ON groundwater_data (district);
CREATE INDEX IF NOT EXISTS idx_gw_status      ON groundwater_data (status);
CREATE INDEX IF NOT EXISTS idx_gw_extr_rate   ON groundwater_data (extraction_rate_pct DESC NULLS LAST);

-- ═══════════════════════════════════════════════════════════
-- TRIGGERS
-- ═══════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_updated_at ON groundwater_data;
CREATE TRIGGER set_updated_at
  BEFORE UPDATE ON groundwater_data
  FOR EACH ROW EXECUTE PROCEDURE trigger_set_updated_at();

DROP TRIGGER IF EXISTS set_updated_at_stations ON stations;
CREATE TRIGGER set_updated_at_stations
  BEFORE UPDATE ON stations
  FOR EACH ROW EXECUTE PROCEDURE trigger_set_updated_at();

DROP TRIGGER IF EXISTS set_updated_at_assessment_units ON assessment_units;
CREATE TRIGGER set_updated_at_assessment_units
  BEFORE UPDATE ON assessment_units
  FOR EACH ROW EXECUTE PROCEDURE trigger_set_updated_at();

DROP TRIGGER IF EXISTS set_updated_at_assessment_records ON assessment_records;
CREATE TRIGGER set_updated_at_assessment_records
  BEFORE UPDATE ON assessment_records
  FOR EACH ROW EXECUTE PROCEDURE trigger_set_updated_at();

DROP TRIGGER IF EXISTS set_updated_at_model_metadata ON model_metadata;
CREATE TRIGGER set_updated_at_model_metadata
  BEFORE UPDATE ON model_metadata
  FOR EACH ROW EXECUTE PROCEDURE trigger_set_updated_at();
