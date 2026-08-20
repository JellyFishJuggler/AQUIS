const { query } = require("../db/pool");

async function createModel(data) {
  const result = await query(
    `INSERT INTO model_metadata (
      task, model_name, model_type, version,
      training_start, training_end, evaluation_start, evaluation_end,
      features, feature_importance, metrics, is_selected, status, artifact_path, uncertainty_config
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
    ON CONFLICT (model_name, version) DO UPDATE SET
      metrics = EXCLUDED.metrics, status = EXCLUDED.status,
      feature_importance = EXCLUDED.feature_importance, updated_at = NOW()
    RETURNING id`,
    [
      data.task, data.model_name, data.model_type, data.version,
      data.training_start || null, data.training_end || null,
      data.evaluation_start || null, data.evaluation_end || null,
      data.features ? JSON.stringify(data.features) : null,
      data.feature_importance ? JSON.stringify(data.feature_importance) : null,
      data.metrics ? JSON.stringify(data.metrics) : null,
      data.is_selected || false, data.status || "trained",
      data.artifact_path || null,
      data.uncertainty_config ? JSON.stringify(data.uncertainty_config) : null,
    ]
  );
  return result.rows[0].id;
}

async function findAll({ task, limit = 50, offset = 0 } = {}) {
  const clauses = [];
  const values = [];
  let idx = 1;
  if (task) { clauses.push(`task = $${idx++}`); values.push(task); }
  const where = clauses.length ? "WHERE " + clauses.join(" AND ") : "";
  const [data, count] = await Promise.all([
    query(`SELECT * FROM model_metadata ${where} ORDER BY created_at DESC LIMIT $${idx++} OFFSET $${idx++}`, [...values, limit, offset]),
    query(`SELECT COUNT(*) AS total FROM model_metadata ${where}`, values),
  ]);
  return { rows: data.rows, total: parseInt(count.rows[0].total, 10) };
}

async function findById(id) {
  const result = await query("SELECT * FROM model_metadata WHERE id = $1", [id]);
  return result.rows[0] || null;
}

async function findByNameAndVersion(name, version) {
  const result = await query(
    "SELECT * FROM model_metadata WHERE model_name = $1 AND version = $2",
    [name, version]
  );
  return result.rows[0] || null;
}

async function getSelected(task) {
  const result = await query(
    "SELECT * FROM model_metadata WHERE task = $1 AND is_selected = TRUE LIMIT 1",
    [task]
  );
  return result.rows[0] || null;
}

async function selectModel(id, task) {
  await query("UPDATE model_metadata SET is_selected = FALSE WHERE task = $1", [task]);
  await query("UPDATE model_metadata SET is_selected = TRUE, status = 'selected' WHERE id = $1", [id]);
}

async function getComparison(task) {
  const result = await query(
    `SELECT model_name, model_type, version, metrics, is_selected, status, created_at,
            training_start, training_end, evaluation_start, evaluation_end, features
     FROM model_metadata
     WHERE task = $1
     ORDER BY created_at DESC`,
    [task]
  );
  return result.rows;
}

async function insertOutput(data) {
  const result = await query(
    `INSERT INTO model_outputs (
      model_id, output_type, entity_type, entity_id,
      predicted_at, target_time, target_year,
      prediction, lower_bound, upper_bound,
      probability, confidence, is_anomaly, anomaly_score,
      risk_category, details
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
    RETURNING id`,
    [
      data.model_id, data.output_type, data.entity_type, data.entity_id,
      data.predicted_at, data.target_time || null, data.target_year || null,
      data.prediction || null, data.lower_bound || null, data.upper_bound || null,
      data.probability || null, data.confidence || null,
      data.is_anomaly || null, data.anomaly_score || null,
      data.risk_category || null,
      data.details ? JSON.stringify(data.details) : null,
    ]
  );
  return result.rows[0].id;
}

async function getOutputs({ modelId, outputType, entityType, entityId, limit = 100, offset = 0 } = {}) {
  const clauses = [];
  const values = [];
  let idx = 1;
  if (modelId) { clauses.push(`model_id = $${idx++}`); values.push(modelId); }
  if (outputType) { clauses.push(`output_type = $${idx++}`); values.push(outputType); }
  if (entityType) { clauses.push(`entity_type = $${idx++}`); values.push(entityType); }
  if (entityId) { clauses.push(`entity_id = $${idx++}`); values.push(entityId); }
  const where = clauses.length ? "WHERE " + clauses.join(" AND ") : "";
  const [data, count] = await Promise.all([
    query(`SELECT mo.*, mm.model_name, mm.model_type, mm.version AS model_version
           FROM model_outputs mo JOIN model_metadata mm ON mo.model_id = mm.id
           ${where} ORDER BY mo.created_at DESC LIMIT $${idx++} OFFSET $${idx++}`, [...values, limit, offset]),
    query(`SELECT COUNT(*) AS total FROM model_outputs mo ${where}`, values),
  ]);
  return { rows: data.rows, total: parseInt(count.rows[0].total, 10) };
}

async function getLatestForecast(stationId) {
  const result = await query(
    `SELECT mo.*, mm.model_name, mm.model_type, mm.version AS model_version, mm.metrics
     FROM model_outputs mo JOIN model_metadata mm ON mo.model_id = mm.id
     WHERE mo.output_type = 'forecast' AND mo.entity_type = 'station' AND mo.entity_id = $1
     ORDER BY mo.predicted_at DESC LIMIT 1`,
    [stationId]
  );
  return result.rows[0] || null;
}

async function getStationAnomalies(stationId, { limit = 50, offset = 0 } = {}) {
  const [data, count] = await Promise.all([
    query(
      `SELECT mo.*, mm.model_name, mm.model_type
       FROM model_outputs mo JOIN model_metadata mm ON mo.model_id = mm.id
       WHERE mo.output_type = 'anomaly' AND mo.entity_type = 'station' AND mo.entity_id = $1
       ORDER BY mo.created_at DESC LIMIT $2 OFFSET $3`,
      [stationId, limit, offset]
    ),
    query(
      `SELECT COUNT(*) AS total FROM model_outputs
       WHERE output_type = 'anomaly' AND entity_type = 'station' AND entity_id = $1`,
      [stationId]
    ),
  ]);
  return { rows: data.rows, total: parseInt(count.rows[0].total, 10) };
}

async function getAnomalySummary() {
  const result = await query(
    `SELECT
       entity_id,
       COUNT(*) AS total_anomalies,
       COUNT(*) FILTER (WHERE is_anomaly = TRUE) AS confirmed_anomalies,
       MAX(created_at) AS latest_detection,
       AVG(anomaly_score) AS avg_anomaly_score
     FROM model_outputs
     WHERE output_type = 'anomaly'
     GROUP BY entity_id
     ORDER BY total_anomalies DESC`
  );
  return result.rows;
}

module.exports = {
  createModel, findAll, findById, findByNameAndVersion,
  getSelected, selectModel, getComparison,
  insertOutput, getOutputs, getLatestForecast,
  getStationAnomalies, getAnomalySummary,
};
