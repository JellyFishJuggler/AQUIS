const { query, pool } = require("../db/pool");

async function runInTransaction(fn) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const result = await fn(client);
    await client.query("COMMIT");
    return result;
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();
  }
}

async function createRun(source, sourceRef, metadata) {
  const result = await query(
    `INSERT INTO ingestion_runs (source, source_ref, metadata)
     VALUES ($1, $2, $3) RETURNING id`,
    [source, sourceRef || null, metadata ? JSON.stringify(metadata) : null]
  );
  return result.rows[0].id;
}

async function finishRun(id, stats) {
  await query(
    `UPDATE ingestion_runs SET
       status = $2, finished_at = NOW(),
       records_seen = $3, records_inserted = $4, records_updated = $5,
       records_rejected = $6, records_duplicates = $7,
       error_count = $8, error_summary = $9
     WHERE id = $1`,
    [
      id,
      stats.status || "completed",
      stats.recordsSeen || 0,
      stats.recordsInserted || 0,
      stats.recordsUpdated || 0,
      stats.recordsRejected || 0,
      stats.recordsDuplicates || 0,
      stats.errorCount || 0,
      stats.errorSummary ? JSON.stringify(stats.errorSummary) : null,
    ]
  );
}

async function getRuns({ source, status, limit = 20, offset = 0 } = {}) {
  const clauses = [];
  const values = [];
  let idx = 1;
  if (source) { clauses.push(`source = $${idx++}`); values.push(source); }
  if (status) { clauses.push(`status = $${idx++}`); values.push(status); }
  const where = clauses.length ? "WHERE " + clauses.join(" AND ") : "";
  const [data, count] = await Promise.all([
    query(`SELECT * FROM ingestion_runs ${where} ORDER BY id DESC LIMIT $${idx++} OFFSET $${idx++}`, [...values, limit, offset]),
    query(`SELECT COUNT(*) AS total FROM ingestion_runs ${where}`, values),
  ]);
  return { rows: data.rows, total: parseInt(count.rows[0].total, 10) };
}

async function getRunById(id) {
  const result = await query("SELECT * FROM ingestion_runs WHERE id = $1", [id]);
  return result.rows[0] || null;
}

module.exports = { runInTransaction, createRun, finishRun, getRuns, getRunById };
