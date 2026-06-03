-- AQUIS Database Schema
-- Run this once to initialise the database.
-- Based on CGWB DWLR Dataset structure from the research paper.

-- ─────────────────────────────────────────────
-- ENUM for stress categories (CGWB classification)
-- ─────────────────────────────────────────────
DO $$ BEGIN
  CREATE TYPE stress_category AS ENUM (
    'SAFE',
    'SEMI_CRITICAL',
    'CRITICAL',
    'OVER_EXPLOITED'
  );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ─────────────────────────────────────────────
-- Main groundwater assessment table
-- Maps directly to CGWB DWLR 2023 dataset
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS groundwater_data (
  id                          SERIAL PRIMARY KEY,

  -- Location
  state                       TEXT NOT NULL,
  district                    TEXT NOT NULL,
  assessment_unit             TEXT,

  -- Rainfall (mm)
  rainfall_mm                 NUMERIC(10, 2),

  -- Recharge components (hectare-metres)
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
  annual_recharge_ham         NUMERIC(14, 4),    -- Annual Ground Water Recharge (Total)

  -- Environmental flows & availability
  environmental_flows_ham     NUMERIC(14, 4),
  annual_extractable_gw_ham   NUMERIC(14, 4),    -- Annual Extractable GW Resource

  -- Extraction breakdown (ha.m)
  extraction_domestic_ham     NUMERIC(14, 4),
  extraction_industrial_ham   NUMERIC(14, 4),
  extraction_irrigation_ham   NUMERIC(14, 4),
  total_extraction_ham        NUMERIC(14, 4),    -- Total GW Extraction (all uses)

  -- Derived metrics
  extraction_rate_pct         NUMERIC(8, 4),     -- Stage of GW Extraction (%)
  net_availability_future_ham NUMERIC(14, 4),    -- Net Annual GW Availability for Future Use

  -- Aquifer storage data
  storage_unconfined_fresh_ham   NUMERIC(14, 4),
  availability_unconfined_fresh  NUMERIC(14, 4),
  availability_confined_fresh    NUMERIC(14, 4),
  availability_semiconfined_fresh NUMERIC(14, 4),
  total_gw_availability_ham      NUMERIC(14, 4), -- Total GW availability in area (fresh)

  -- Computed classification (derived by application logic, not stored as user input)
  status                      stress_category,

  -- Meta
  created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────
-- Indexes for common query patterns
-- ─────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_gw_state       ON groundwater_data (state);
CREATE INDEX IF NOT EXISTS idx_gw_district    ON groundwater_data (district);
CREATE INDEX IF NOT EXISTS idx_gw_status      ON groundwater_data (status);
CREATE INDEX IF NOT EXISTS idx_gw_extr_rate   ON groundwater_data (extraction_rate_pct DESC NULLS LAST);

-- ─────────────────────────────────────────────
-- Auto-update updated_at on row change
-- ─────────────────────────────────────────────
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
