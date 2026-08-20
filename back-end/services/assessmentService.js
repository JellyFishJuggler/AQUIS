const { query } = require("../db/pool");

function buildExternalUnitId(state, district, assessmentUnit) {
  return `${(state || "").trim()}|${(district || "").trim()}|${(assessmentUnit || "").trim()}`.toUpperCase();
}

async function upsertUnit(state, district, assessmentUnit) {
  const externalId = buildExternalUnitId(state, district, assessmentUnit);
  const result = await query(
    `INSERT INTO assessment_units (external_unit_id, state, district, assessment_unit)
     VALUES ($1, $2, $3, $4)
     ON CONFLICT (external_unit_id) DO UPDATE SET updated_at = NOW()
     RETURNING id`,
    [externalId, (state || "").trim(), (district || "").trim(), (assessmentUnit || "").trim() || null]
  );
  return result.rows[0].id;
}

async function findUnits({ state, district, limit = 50, offset = 0 } = {}) {
  const clauses = [];
  const values = [];
  let idx = 1;
  if (state) { clauses.push(`LOWER(state) = LOWER($${idx++})`); values.push(state); }
  if (district) { clauses.push(`LOWER(district) = LOWER($${idx++})`); values.push(district); }
  const where = clauses.length ? "WHERE " + clauses.join(" AND ") : "";
  const [data, count] = await Promise.all([
    query(`SELECT * FROM assessment_units ${where} ORDER BY id LIMIT $${idx++} OFFSET $${idx++}`, [...values, limit, offset]),
    query(`SELECT COUNT(*) AS total FROM assessment_units ${where}`, values),
  ]);
  return { rows: data.rows, total: parseInt(count.rows[0].total, 10) };
}

async function findUnitById(id) {
  const result = await query("SELECT * FROM assessment_units WHERE id = $1", [id]);
  return result.rows[0] || null;
}

async function insertRecord(data) {
  const result = await query(
    `INSERT INTO assessment_records (
      assessment_unit_id, assessment_year, rainfall_mm,
      recharge_worthy_area_ha, recharge_rainfall_ham, recharge_canals_ham,
      recharge_surface_irr_ham, recharge_gw_irr_ham, recharge_tanks_ponds_ham,
      recharge_wcs_ham, recharge_pipelines_ham, recharge_sewage_ff_ham,
      total_recharge_ham, stream_recharge_ham, annual_recharge_ham,
      environmental_flows_ham, annual_extractable_gw_ham,
      extraction_domestic_ham, extraction_industrial_ham, extraction_irrigation_ham,
      total_extraction_ham, stage_of_extraction_pct, extraction_rate_pct,
      net_availability_future_ham, allocation_domestic_ham,
      quality_tagging_major, quality_tagging_other, coastal_area_ham,
      storage_unconfined_fresh, storage_unconfined_saline,
      availability_unconfined_fresh, availability_unconfined_saline,
      dynamic_confined_fresh, dynamic_confined_saline,
      storage_confined_fresh, storage_confined_saline,
      total_confined_fresh, total_confined_saline,
      dynamic_semiconfined_fresh, dynamic_semiconfined_saline,
      storage_semiconfined_fresh, storage_semiconfined_saline,
      total_semiconfined_fresh, total_semiconfined_saline,
      total_gw_availability_fresh, total_gw_availability_saline,
      total_gw_availability_ham, total_geographical_area_ha, hilly_area_ha,
      status, raw_data, source, source_file, source_record_id, ingestion_run_id
    ) VALUES (
      $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,
      $17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,
      $31,$32,$33,$34,$35,$36,$37,$38,$39,$40,$41,$42,$43,$44,
      $45,$46,$47,$48,$49,$50,$51,$52,$53,$54,$55
    )
    ON CONFLICT (assessment_unit_id, assessment_year) DO UPDATE SET
      rainfall_mm = EXCLUDED.rainfall_mm,
      recharge_worthy_area_ha = EXCLUDED.recharge_worthy_area_ha,
      total_recharge_ham = EXCLUDED.total_recharge_ham,
      annual_recharge_ham = EXCLUDED.annual_recharge_ham,
      annual_extractable_gw_ham = EXCLUDED.annual_extractable_gw_ham,
      total_extraction_ham = EXCLUDED.total_extraction_ham,
      extraction_rate_pct = EXCLUDED.extraction_rate_pct,
      stage_of_extraction_pct = EXCLUDED.stage_of_extraction_pct,
      status = EXCLUDED.status,
      raw_data = EXCLUDED.raw_data,
      source_file = EXCLUDED.source_file,
      ingestion_run_id = EXCLUDED.ingestion_run_id,
      updated_at = NOW()
    RETURNING id`,
    [
      data.assessment_unit_id, data.assessment_year, data.rainfall_mm,
      data.recharge_worthy_area_ha, data.recharge_rainfall_ham, data.recharge_canals_ham,
      data.recharge_surface_irr_ham, data.recharge_gw_irr_ham, data.recharge_tanks_ponds_ham,
      data.recharge_wcs_ham, data.recharge_pipelines_ham, data.recharge_sewage_ff_ham,
      data.total_recharge_ham, data.stream_recharge_ham, data.annual_recharge_ham,
      data.environmental_flows_ham, data.annual_extractable_gw_ham,
      data.extraction_domestic_ham, data.extraction_industrial_ham, data.extraction_irrigation_ham,
      data.total_extraction_ham, data.stage_of_extraction_pct, data.extraction_rate_pct,
      data.net_availability_future_ham, data.allocation_domestic_ham,
      data.quality_tagging_major, data.quality_tagging_other, data.coastal_area_ham,
      data.storage_unconfined_fresh, data.storage_unconfined_saline,
      data.availability_unconfined_fresh, data.availability_unconfined_saline,
      data.dynamic_confined_fresh, data.dynamic_confined_saline,
      data.storage_confined_fresh, data.storage_confined_saline,
      data.total_confined_fresh, data.total_confined_saline,
      data.dynamic_semiconfined_fresh, data.dynamic_semiconfined_saline,
      data.storage_semiconfined_fresh, data.storage_semiconfined_saline,
      data.total_semiconfined_fresh, data.total_semiconfined_saline,
      data.total_gw_availability_fresh, data.total_gw_availability_saline,
      data.total_gw_availability_ham, data.total_geographical_area_ha, data.hilly_area_ha,
      data.status, data.raw_data ? JSON.stringify(data.raw_data) : null,
      data.source || "centralreport_ingres", data.source_file || null,
      data.source_record_id || null, data.ingestion_run_id || null,
    ]
  );
  return result.rows[0].id;
}

async function findRecords({ state, district, year, status, unitId, limit = 50, offset = 0 } = {}) {
  const clauses = [];
  const values = [];
  let idx = 1;

  if (unitId) { clauses.push(`ar.assessment_unit_id = $${idx++}`); values.push(unitId); }
  if (year) { clauses.push(`ar.assessment_year = $${idx++}`); values.push(year); }
  if (status) { clauses.push(`ar.status = $${idx++}`); values.push(status); }
  if (state) { clauses.push(`LOWER(au.state) = LOWER($${idx++})`); values.push(state); }
  if (district) { clauses.push(`LOWER(au.district) = LOWER($${idx++})`); values.push(district); }

  const where = clauses.length ? "WHERE " + clauses.join(" AND ") : "";
  const [data, count] = await Promise.all([
    query(
      `SELECT ar.*, au.state, au.district, au.assessment_unit, au.external_unit_id
       FROM assessment_records ar
       JOIN assessment_units au ON ar.assessment_unit_id = au.id
       ${where}
       ORDER BY ar.assessment_year DESC, au.state, au.district
       LIMIT $${idx++} OFFSET $${idx++}`,
      [...values, limit, offset]
    ),
    query(
      `SELECT COUNT(*) AS total
       FROM assessment_records ar
       JOIN assessment_units au ON ar.assessment_unit_id = au.id
       ${where}`,
      values
    ),
  ]);
  return { rows: data.rows, total: parseInt(count.rows[0].total, 10) };
}

async function findRecordById(id) {
  const result = await query(
    `SELECT ar.*, au.state, au.district, au.assessment_unit, au.external_unit_id
     FROM assessment_records ar
     JOIN assessment_units au ON ar.assessment_unit_id = au.id
     WHERE ar.id = $1`,
    [id]
  );
  return result.rows[0] || null;
}

async function getUnitHistory(unitId) {
  const result = await query(
    `SELECT ar.*, au.state, au.district, au.assessment_unit
     FROM assessment_records ar
     JOIN assessment_units au ON ar.assessment_unit_id = au.id
     WHERE ar.assessment_unit_id = $1
     ORDER BY ar.assessment_year ASC`,
    [unitId]
  );
  return result.rows;
}

async function getSummary({ state, year } = {}) {
  const clauses = [];
  const values = [];
  let idx = 1;
  if (state) { clauses.push(`LOWER(au.state) = LOWER($${idx++})`); values.push(state); }
  if (year) { clauses.push(`ar.assessment_year = $${idx++}`); values.push(year); }
  const where = clauses.length ? "WHERE " + clauses.join(" AND ") : "";

  const result = await query(
    `SELECT
       au.state,
       ar.assessment_year,
       COUNT(*) AS unit_count,
       ROUND(AVG(ar.recharge_rainfall_ham)::numeric, 4) AS avg_recharge,
       ROUND(AVG(ar.total_extraction_ham)::numeric, 4) AS avg_extraction,
       ROUND(AVG(ar.annual_extractable_gw_ham)::numeric, 4) AS avg_extractable,
       ROUND(AVG(ar.stage_of_extraction_pct)::numeric, 2) AS avg_stage_of_extraction,
       COUNT(*) FILTER (WHERE ar.status = 'SAFE') AS count_safe,
       COUNT(*) FILTER (WHERE ar.status = 'SEMI_CRITICAL') AS count_semi_critical,
       COUNT(*) FILTER (WHERE ar.status = 'CRITICAL') AS count_critical,
       COUNT(*) FILTER (WHERE ar.status = 'OVER_EXPLOITED') AS count_over_exploited
     FROM assessment_records ar
     JOIN assessment_units au ON ar.assessment_unit_id = au.id
     ${where}
     GROUP BY au.state, ar.assessment_year
     ORDER BY au.state, ar.assessment_year`,
    values
  );
  return result.rows;
}

async function getAvailableYears() {
  const result = await query(
    `SELECT DISTINCT assessment_year, COUNT(*) AS record_count
     FROM assessment_records
     GROUP BY assessment_year
     ORDER BY assessment_year DESC`
  );
  return result.rows;
}

module.exports = {
  buildExternalUnitId, upsertUnit, findUnits, findUnitById,
  insertRecord, findRecords, findRecordById, getUnitHistory,
  getSummary, getAvailableYears,
};
