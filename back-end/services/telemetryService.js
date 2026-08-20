const { query } = require("../db/pool");

async function insertObservation(data) {
  const result = await query(
    `INSERT INTO telemetry_observations
       (station_id, observed_at, groundwater_level, source, source_record_id, ingestion_run_id)
     VALUES ($1, $2, $3, $4, $5, $6)
     ON CONFLICT (station_id, observed_at) DO NOTHING
     RETURNING id`,
    [data.station_id, data.observed_at, data.groundwater_level,
     data.source || "nwdp_api", data.source_record_id || null, data.ingestion_run_id || null]
  );
  return result.rows[0] ? result.rows[0].id : null;
}

async function insertBatch(observations) {
  let inserted = 0;
  let duplicates = 0;
  for (const obs of observations) {
    const id = await insertObservation(obs);
    if (id) inserted++; else duplicates++;
  }
  return { inserted, duplicates };
}

async function findByStationId(stationId, { startDate, endDate, limit = 1000, offset = 0 } = {}) {
  const clauses = ["station_id = $1"];
  const values = [stationId];
  let idx = 2;
  if (startDate) { clauses.push(`observed_at >= $${idx++}`); values.push(startDate); }
  if (endDate) { clauses.push(`observed_at <= $${idx++}`); values.push(endDate); }
  const where = clauses.join(" AND ");
  const [data, count] = await Promise.all([
    query(`SELECT * FROM telemetry_observations WHERE ${where} ORDER BY observed_at ASC LIMIT $${idx++} OFFSET $${idx++}`, [...values, limit, offset]),
    query(`SELECT COUNT(*) AS total FROM telemetry_observations WHERE ${where}`, values),
  ]);
  return { rows: data.rows, total: parseInt(count.rows[0].total, 10) };
}

async function getLatestByStation(stationId) {
  const result = await query(
    `SELECT * FROM telemetry_observations
     WHERE station_id = $1 ORDER BY observed_at DESC LIMIT 1`,
    [stationId]
  );
  return result.rows[0] || null;
}

async function findMany({ state, district, stationId, startDate, endDate, limit = 100, offset = 0 } = {}) {
  let joinClause = "";
  const clauses = [];
  const values = [];
  let idx = 1;

  if (state) { joinClause = "JOIN stations s ON t.station_id = s.id"; clauses.push(`LOWER(s.state) = LOWER($${idx++})`); values.push(state); }
  if (district) { if (!joinClause) joinClause = "JOIN stations s ON t.station_id = s.id"; clauses.push(`LOWER(s.district) = LOWER($${idx++})`); values.push(district); }
  if (stationId) { clauses.push(`t.station_id = $${idx++}`); values.push(stationId); }
  if (startDate) { clauses.push(`t.observed_at >= $${idx++}`); values.push(startDate); }
  if (endDate) { clauses.push(`t.observed_at <= $${idx++}`); values.push(endDate); }

  const where = clauses.length ? "WHERE " + clauses.join(" AND ") : "";
  const [data, count] = await Promise.all([
    query(`SELECT t.*, s.station_name, s.state, s.district, s.latitude, s.longitude
           FROM telemetry_observations t ${joinClause} ${where}
           ORDER BY t.observed_at ASC LIMIT $${idx++} OFFSET $${idx++}`, [...values, limit, offset]),
    query(`SELECT COUNT(*) AS total FROM telemetry_observations t ${joinClause} ${where}`, values),
  ]);
  return { rows: data.rows, total: parseInt(count.rows[0].total, 10) };
}

async function getStateDistrictSummary() {
  const result = await query(
    `SELECT s.state, s.district,
       COUNT(DISTINCT t.station_id) AS station_count,
       COUNT(*) AS observation_count,
       MIN(t.observed_at) AS earliest_observation,
       MAX(t.observed_at) AS latest_observation,
       ROUND(AVG(t.groundwater_level)::numeric, 4) AS avg_groundwater_level
     FROM telemetry_observations t
     JOIN stations s ON t.station_id = s.id
     GROUP BY s.state, s.district
     ORDER BY observation_count DESC`
  );
  return result.rows;
}

module.exports = {
  insertObservation, insertBatch, findByStationId,
  getLatestByStation, findMany, getStateDistrictSummary,
};
