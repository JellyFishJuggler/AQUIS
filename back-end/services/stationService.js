const { query } = require("../db/pool");

async function upsertStation(data) {
  const result = await query(
    `INSERT INTO stations (
      external_station_id, station_name, agency, state, state_lgd_code,
      district, district_lgd_code, tehsil, block, village,
      river, basin, tributary, subtributary, sub_subtributary,
      local_river, latitude, longitude, rl_msl
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
    ON CONFLICT (external_station_id) DO UPDATE SET
      station_name = COALESCE(EXCLUDED.station_name, stations.station_name),
      agency = COALESCE(EXCLUDED.agency, stations.agency),
      state = COALESCE(EXCLUDED.state, stations.state),
      state_lgd_code = COALESCE(EXCLUDED.state_lgd_code, stations.state_lgd_code),
      district = COALESCE(EXCLUDED.district, stations.district),
      district_lgd_code = COALESCE(EXCLUDED.district_lgd_code, stations.district_lgd_code),
      tehsil = COALESCE(EXCLUDED.tehsil, stations.tehsil),
      block = COALESCE(EXCLUDED.block, stations.block),
      village = COALESCE(EXCLUDED.village, stations.village),
      river = COALESCE(EXCLUDED.river, stations.river),
      basin = COALESCE(EXCLUDED.basin, stations.basin),
      tributary = COALESCE(EXCLUDED.tributary, stations.tributary),
      subtributary = COALESCE(EXCLUDED.subtributary, stations.subtributary),
      sub_subtributary = COALESCE(EXCLUDED.sub_subtributary, stations.sub_subtributary),
      local_river = COALESCE(EXCLUDED.local_river, stations.local_river),
      latitude = COALESCE(EXCLUDED.latitude, stations.latitude),
      longitude = COALESCE(EXCLUDED.longitude, stations.longitude),
      rl_msl = COALESCE(EXCLUDED.rl_msl, stations.rl_msl),
      updated_at = NOW()
    RETURNING id`,
    [
      data.external_station_id, data.station_name, data.agency,
      data.state, data.state_lgd_code, data.district, data.district_lgd_code,
      data.tehsil, data.block, data.village,
      data.river, data.basin, data.tributary, data.subtributary,
      data.sub_subtributary, data.local_river,
      data.latitude, data.longitude, data.rl_msl,
    ]
  );
  return result.rows[0].id;
}

async function updateStationObsStats(stationId, observedAt) {
  await query(
    `UPDATE stations SET
       first_observed_at = LEAST(COALESCE(first_observed_at, $2), $2),
       last_observed_at = GREATEST(COALESCE(last_observed_at, $2), $2),
       observation_count = observation_count + 1
     WHERE id = $1`,
    [stationId, observedAt]
  );
}

async function findAll({ state, district, agency, limit = 50, offset = 0 } = {}) {
  const clauses = [];
  const values = [];
  let idx = 1;
  if (state) { clauses.push(`LOWER(state) = LOWER($${idx++})`); values.push(state); }
  if (district) { clauses.push(`LOWER(district) = LOWER($${idx++})`); values.push(district); }
  if (agency) { clauses.push(`LOWER(agency) = LOWER($${idx++})`); values.push(agency); }
  const where = clauses.length ? "WHERE " + clauses.join(" AND ") : "";
  const [data, count] = await Promise.all([
    query(`SELECT * FROM stations ${where} ORDER BY id LIMIT $${idx++} OFFSET $${idx++}`, [...values, limit, offset]),
    query(`SELECT COUNT(*) AS total FROM stations ${where}`, values),
  ]);
  return { rows: data.rows, total: parseInt(count.rows[0].total, 10) };
}

async function findById(id) {
  const result = await query("SELECT * FROM stations WHERE id = $1", [id]);
  return result.rows[0] || null;
}

async function findByExternalId(externalId) {
  const result = await query("SELECT * FROM stations WHERE external_station_id = $1", [externalId]);
  return result.rows[0] || null;
}

async function findNearby(lat, lon, radiusKm = 50, limit = 20) {
  const result = await query(
    `SELECT *, (
       6371 * acos(
         cos(radians($1)) * cos(radians(latitude)) *
         cos(radians(longitude) - radians($2)) +
         sin(radians($1)) * sin(radians(latitude))
       )
     ) AS distance_km
     FROM stations
     WHERE latitude IS NOT NULL AND longitude IS NOT NULL
     ORDER BY distance_km ASC
     LIMIT $3`,
    [lat, lon, limit]
  );
  return result.rows;
}

async function getStateSummary() {
  const result = await query(
    `SELECT state, COUNT(*) AS station_count,
       MIN(first_observed_at) AS earliest_observation,
       MAX(last_observed_at) AS latest_observation
     FROM stations
     WHERE state IS NOT NULL
     GROUP BY state
     ORDER BY station_count DESC`
  );
  return result.rows;
}

async function getDistrictSummary(state) {
  let sql = `SELECT state, district, COUNT(*) AS station_count,
    MIN(first_observed_at) AS earliest_observation,
    MAX(last_observed_at) AS latest_observation
    FROM stations WHERE district IS NOT NULL`;
  const values = [];
  if (state) {
    values.push(state);
    sql += ` AND LOWER(state) = LOWER($1)`;
  }
  sql += ` GROUP BY state, district ORDER BY station_count DESC`;
  const result = await query(sql, values);
  return result.rows;
}

module.exports = {
  upsertStation, updateStationObsStats,
  findAll, findById, findByExternalId, findNearby,
  getStateSummary, getDistrictSummary,
};
