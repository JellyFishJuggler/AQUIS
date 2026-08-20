const { query } = require("../db/pool");

async function runCheck(checkName, entityType, entityId, severity, message, details) {
  await query(
    `INSERT INTO data_quality (entity_type, entity_id, check_name, severity, message, details)
     VALUES ($1, $2, $3, $4, $5, $6)`,
    [entityType, entityId, checkName, severity || "warning", message || null, details ? JSON.stringify(details) : null]
  );
}

async function checkStationCoordinates(stationId, lat, lon) {
  if (lat == null || lon == null) {
    await runCheck("missing_coordinates", "station", stationId, "warning", "Station has no coordinates");
    return false;
  }
  if (lat < 6 || lat > 38 || lon < 68 || lon > 98) {
    await runCheck("invalid_coordinates", "station", stationId, "error", `Coordinates out of India bounds: ${lat}, ${lon}`);
    return false;
  }
  return true;
}

async function checkTelemetryObservation(obs) {
  const issues = [];
  if (!obs.observed_at) issues.push({ check: "missing_timestamp", severity: "error" });
  if (obs.groundwater_level == null) issues.push({ check: "missing_groundwater_level", severity: "warning" });
  if (obs.groundwater_level != null && (obs.groundwater_level > 100 || obs.groundwater_level < -200)) {
    issues.push({ check: "implausible_groundwater_level", severity: "error", value: obs.groundwater_level });
  }
  return issues;
}

async function checkAssessmentRecord(record) {
  const issues = [];
  if (!record.assessment_year) issues.push({ check: "missing_year", severity: "error" });
  if (record.total_extraction_ham != null && record.total_extraction_ham < 0) {
    issues.push({ check: "negative_extraction", severity: "error", value: record.total_extraction_ham });
  }
  if (record.annual_extractable_gw_ham != null && record.annual_extractable_gw_ham < 0) {
    issues.push({ check: "negative_extractable", severity: "error", value: record.annual_extractable_gw_ham });
  }
  if (record.rainfall_mm != null && record.rainfall_mm < 0) {
    issues.push({ check: "negative_rainfall", severity: "error", value: record.rainfall_mm });
  }
  if (record.status == null && record.stage_of_extraction_pct != null) {
    issues.push({ check: "missing_status", severity: "warning" });
  }
  return issues;
}

async function getStationQuality(stationId) {
  const result = await query(
    `SELECT * FROM data_quality
     WHERE entity_type = 'station' AND entity_id = $1
     ORDER BY created_at DESC`,
    [stationId]
  );
  return result.rows;
}

async function getTelemetryQualitySummary() {
  const result = await query(
    `SELECT check_name, severity, COUNT(*) AS count
     FROM data_quality
     WHERE entity_type = 'telemetry'
     GROUP BY check_name, severity
     ORDER BY count DESC`
  );
  return result.rows;
}

async function getAssessmentQualitySummary(year) {
  const clauses = ["entity_type = 'assessment'"];
  const values = [];
  let idx = 1;
  if (year) { clauses.push(`details->>'year' = $${idx++}`); values.push(year); }
  const where = "WHERE " + clauses.join(" AND ");
  const result = await query(
    `SELECT check_name, severity, COUNT(*) AS count
     FROM data_quality ${where}
     GROUP BY check_name, severity
     ORDER BY count DESC`,
    values
  );
  return result.rows;
}

module.exports = {
  runCheck, checkStationCoordinates, checkTelemetryObservation,
  checkAssessmentRecord, getStationQuality, getTelemetryQualitySummary,
  getAssessmentQualitySummary,
};
