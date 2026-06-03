/**
 * groundwaterRepository.js
 *
 * All raw SQL for the groundwater_data table.
 * Controllers never touch SQL directly — they go through this module.
 */

const { query } = require("../db/pool");
const { deriveClassification } = require("./groundwaterClassification");

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────

/**
 * Build a dynamic WHERE clause from a filter object.
 * Allowed filter keys are whitelisted here for safety.
 */
function buildWhereClause(filters = {}) {
  const allowed = ["state", "district", "status", "assessment_unit"];
  const clauses = [];
  const values = [];
  let idx = 1;

  for (const [key, val] of Object.entries(filters)) {
    if (!allowed.includes(key) || val == null || val === "") continue;
    clauses.push(`LOWER(${key}::text) = LOWER($${idx})`);
    values.push(String(val));
    idx++;
  }

  return {
    where: clauses.length ? "WHERE " + clauses.join(" AND ") : "",
    values,
  };
}

// ─────────────────────────────────────────────────────────
// READ
// ─────────────────────────────────────────────────────────

/**
 * Paginated list with optional filters.
 *
 * @param {object} options
 * @param {number} options.limit
 * @param {number} options.offset
 * @param {string} [options.state]
 * @param {string} [options.district]
 * @param {string} [options.status]
 * @param {string} [options.sort_by]  – column name
 * @param {string} [options.order]    – 'asc' | 'desc'
 * @returns {Promise<{ rows: object[], total: number }>}
 */
async function findAll({
  limit = 50,
  offset = 0,
  state,
  district,
  status,
  sort_by = "id",
  order = "asc",
} = {}) {
  const ALLOWED_SORT = [
    "id", "state", "district", "extraction_rate_pct",
    "total_extraction_ham", "rainfall_mm", "status",
  ];
  const sortCol = ALLOWED_SORT.includes(sort_by) ? sort_by : "id";
  const sortDir = order.toLowerCase() === "desc" ? "DESC" : "ASC";

  const { where, values } = buildWhereClause({ state, district, status });
  const limitIdx = values.length + 1;
  const offsetIdx = values.length + 2;

  const dataQuery = `
    SELECT *
    FROM   groundwater_data
    ${where}
    ORDER  BY ${sortCol} ${sortDir} NULLS LAST
    LIMIT  $${limitIdx} OFFSET $${offsetIdx}
  `;

  const countQuery = `
    SELECT COUNT(*) AS total
    FROM   groundwater_data
    ${where}
  `;

  const [dataResult, countResult] = await Promise.all([
    query(dataQuery, [...values, limit, offset]),
    query(countQuery, values),
  ]);

  return {
    rows: dataResult.rows,
    total: parseInt(countResult.rows[0].total, 10),
  };
}

/**
 * Single record by primary key.
 */
async function findById(id) {
  const result = await query(
    "SELECT * FROM groundwater_data WHERE id = $1",
    [id]
  );
  return result.rows[0] || null;
}

// ─────────────────────────────────────────────────────────
// AGGREGATIONS
// ─────────────────────────────────────────────────────────

/**
 * Per-state summary: avg rainfall, avg extraction rate, count, status breakdown.
 */
async function aggregateByState(stateFilter) {
  const { where, values } = buildWhereClause(
    stateFilter ? { state: stateFilter } : {}
  );

  const result = await query(
    `
    SELECT
      state,
      COUNT(*)                                    AS district_count,
      ROUND(AVG(rainfall_mm)::numeric, 2)         AS avg_rainfall_mm,
      ROUND(AVG(extraction_rate_pct)::numeric, 2) AS avg_extraction_rate_pct,
      ROUND(AVG(total_extraction_ham)::numeric, 4) AS avg_extraction_ham,
      ROUND(AVG(annual_extractable_gw_ham)::numeric, 4) AS avg_extractable_ham,
      ROUND(AVG(total_gw_availability_ham)::numeric, 4) AS avg_gw_availability_ham,
      COUNT(*) FILTER (WHERE status = 'SAFE')           AS count_safe,
      COUNT(*) FILTER (WHERE status = 'SEMI_CRITICAL')  AS count_semi_critical,
      COUNT(*) FILTER (WHERE status = 'CRITICAL')       AS count_critical,
      COUNT(*) FILTER (WHERE status = 'OVER_EXPLOITED') AS count_over_exploited
    FROM groundwater_data
    ${where}
    GROUP BY state
    ORDER BY avg_extraction_rate_pct DESC NULLS LAST
    `,
    values
  );
  return result.rows;
}

/**
 * Top N hotspot districts by mean extraction rate.
 * Mirrors Table IV from the research paper.
 */
async function getHotspots(limit = 10) {
  const result = await query(
    `
    SELECT
      state,
      district,
      ROUND(AVG(extraction_rate_pct)::numeric, 2) AS mean_extraction_rate_pct,
      status
    FROM groundwater_data
    WHERE extraction_rate_pct IS NOT NULL
    GROUP BY state, district, status
    ORDER BY mean_extraction_rate_pct DESC NULLS LAST
    LIMIT $1
    `,
    [limit]
  );
  return result.rows;
}

/**
 * Overall status distribution — mirrors Table III from the paper.
 */
async function getStatusDistribution() {
  const result = await query(`
    SELECT
      status,
      COUNT(*)                                           AS count,
      ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
    FROM groundwater_data
    WHERE status IS NOT NULL
    GROUP BY status
    ORDER BY
      CASE status
        WHEN 'SAFE'          THEN 1
        WHEN 'SEMI_CRITICAL' THEN 2
        WHEN 'CRITICAL'      THEN 3
        WHEN 'OVER_EXPLOITED' THEN 4
      END
  `);
  return result.rows;
}

// ─────────────────────────────────────────────────────────
// CREATE
// ─────────────────────────────────────────────────────────

/**
 * Insert a single record.
 * Classification is derived server-side; caller must NOT pre-set status.
 */
async function insertOne(data) {
  const { extractionRatePct, status } = deriveClassification(
    data.total_extraction_ham,
    data.annual_extractable_gw_ham
  );

  const result = await query(
    `
    INSERT INTO groundwater_data (
      state, district, assessment_unit,
      rainfall_mm,
      recharge_worthy_area_ha, recharge_rainfall_ham, recharge_canals_ham,
      recharge_surface_irr_ham, recharge_gw_irr_ham, recharge_tanks_ponds_ham,
      recharge_wcs_ham, recharge_pipelines_ham, recharge_sewage_ff_ham,
      total_recharge_ham, stream_recharge_ham, annual_recharge_ham,
      environmental_flows_ham, annual_extractable_gw_ham,
      extraction_domestic_ham, extraction_industrial_ham, extraction_irrigation_ham,
      total_extraction_ham,
      extraction_rate_pct,
      net_availability_future_ham,
      storage_unconfined_fresh_ham, availability_unconfined_fresh,
      availability_confined_fresh, availability_semiconfined_fresh,
      total_gw_availability_ham,
      status
    ) VALUES (
      $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,
      $17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30
    )
    RETURNING *
    `,
    [
      data.state, data.district, data.assessment_unit ?? null,
      data.rainfall_mm ?? null,
      data.recharge_worthy_area_ha ?? null,
      data.recharge_rainfall_ham ?? null,
      data.recharge_canals_ham ?? null,
      data.recharge_surface_irr_ham ?? null,
      data.recharge_gw_irr_ham ?? null,
      data.recharge_tanks_ponds_ham ?? null,
      data.recharge_wcs_ham ?? null,
      data.recharge_pipelines_ham ?? null,
      data.recharge_sewage_ff_ham ?? null,
      data.total_recharge_ham ?? null,
      data.stream_recharge_ham ?? null,
      data.annual_recharge_ham ?? null,
      data.environmental_flows_ham ?? null,
      data.annual_extractable_gw_ham ?? null,
      data.extraction_domestic_ham ?? null,
      data.extraction_industrial_ham ?? null,
      data.extraction_irrigation_ham ?? null,
      data.total_extraction_ham ?? null,
      extractionRatePct,
      data.net_availability_future_ham ?? null,
      data.storage_unconfined_fresh_ham ?? null,
      data.availability_unconfined_fresh ?? null,
      data.availability_confined_fresh ?? null,
      data.availability_semiconfined_fresh ?? null,
      data.total_gw_availability_ham ?? null,
      status,
    ]
  );
  return result.rows[0];
}

/**
 * Bulk insert for the data-pipeline script.
 * Uses a single transaction for atomicity.
 */
async function bulkInsert(records) {
  const { pool } = require("../db/pool");
  const client = await pool.connect();
  let inserted = 0;
  let skipped = 0;

  try {
    await client.query("BEGIN");

    for (const row of records) {
      // Skip rows where both state and district are absent
      if (!row.state || !row.district) {
        skipped++;
        continue;
      }

      const { extractionRatePct, status } = deriveClassification(
        row.total_extraction_ham,
        row.annual_extractable_gw_ham
      );

      await client.query(
        `
        INSERT INTO groundwater_data (
          state, district, assessment_unit,
          rainfall_mm,
          recharge_worthy_area_ha, recharge_rainfall_ham, recharge_canals_ham,
          recharge_surface_irr_ham, recharge_gw_irr_ham, recharge_tanks_ponds_ham,
          recharge_wcs_ham, recharge_pipelines_ham, recharge_sewage_ff_ham,
          total_recharge_ham, stream_recharge_ham, annual_recharge_ham,
          environmental_flows_ham, annual_extractable_gw_ham,
          extraction_domestic_ham, extraction_industrial_ham, extraction_irrigation_ham,
          total_extraction_ham,
          extraction_rate_pct,
          net_availability_future_ham,
          storage_unconfined_fresh_ham, availability_unconfined_fresh,
          availability_confined_fresh, availability_semiconfined_fresh,
          total_gw_availability_ham,
          status
        ) VALUES (
          $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,
          $17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30
        )
        ON CONFLICT DO NOTHING
        `,
        [
          row.state, row.district, row.assessment_unit ?? null,
          row.rainfall_mm ?? null,
          row.recharge_worthy_area_ha ?? null,
          row.recharge_rainfall_ham ?? null,
          row.recharge_canals_ham ?? null,
          row.recharge_surface_irr_ham ?? null,
          row.recharge_gw_irr_ham ?? null,
          row.recharge_tanks_ponds_ham ?? null,
          row.recharge_wcs_ham ?? null,
          row.recharge_pipelines_ham ?? null,
          row.recharge_sewage_ff_ham ?? null,
          row.total_recharge_ham ?? null,
          row.stream_recharge_ham ?? null,
          row.annual_recharge_ham ?? null,
          row.environmental_flows_ham ?? null,
          row.annual_extractable_gw_ham ?? null,
          row.extraction_domestic_ham ?? null,
          row.extraction_industrial_ham ?? null,
          row.extraction_irrigation_ham ?? null,
          row.total_extraction_ham ?? null,
          extractionRatePct,
          row.net_availability_future_ham ?? null,
          row.storage_unconfined_fresh_ham ?? null,
          row.availability_unconfined_fresh ?? null,
          row.availability_confined_fresh ?? null,
          row.availability_semiconfined_fresh ?? null,
          row.total_gw_availability_ham ?? null,
          status,
        ]
      );
      inserted++;
    }

    await client.query("COMMIT");
    return { inserted, skipped };
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();
  }
}

// ─────────────────────────────────────────────────────────
// UPDATE
// ─────────────────────────────────────────────────────────

/**
 * Partial update — only supplied fields are updated.
 * Status is always re-derived after any update.
 */
async function updateOne(id, data) {
  // First fetch existing to fill missing extraction fields for re-classification
  const existing = await findById(id);
  if (!existing) return null;

  const merged = { ...existing, ...data };
  const { extractionRatePct, status } = deriveClassification(
    merged.total_extraction_ham,
    merged.annual_extractable_gw_ham
  );
  merged.extraction_rate_pct = extractionRatePct;
  merged.status = status;

  const result = await query(
    `
    UPDATE groundwater_data SET
      state                       = $1,
      district                    = $2,
      assessment_unit             = $3,
      rainfall_mm                 = $4,
      recharge_worthy_area_ha     = $5,
      recharge_rainfall_ham       = $6,
      recharge_canals_ham         = $7,
      recharge_surface_irr_ham    = $8,
      recharge_gw_irr_ham         = $9,
      recharge_tanks_ponds_ham    = $10,
      recharge_wcs_ham            = $11,
      recharge_pipelines_ham      = $12,
      recharge_sewage_ff_ham      = $13,
      total_recharge_ham          = $14,
      stream_recharge_ham         = $15,
      annual_recharge_ham         = $16,
      environmental_flows_ham     = $17,
      annual_extractable_gw_ham   = $18,
      extraction_domestic_ham     = $19,
      extraction_industrial_ham   = $20,
      extraction_irrigation_ham   = $21,
      total_extraction_ham        = $22,
      extraction_rate_pct         = $23,
      net_availability_future_ham = $24,
      storage_unconfined_fresh_ham  = $25,
      availability_unconfined_fresh = $26,
      availability_confined_fresh   = $27,
      availability_semiconfined_fresh = $28,
      total_gw_availability_ham     = $29,
      status                        = $30
    WHERE id = $31
    RETURNING *
    `,
    [
      merged.state, merged.district, merged.assessment_unit,
      merged.rainfall_mm,
      merged.recharge_worthy_area_ha, merged.recharge_rainfall_ham,
      merged.recharge_canals_ham, merged.recharge_surface_irr_ham,
      merged.recharge_gw_irr_ham, merged.recharge_tanks_ponds_ham,
      merged.recharge_wcs_ham, merged.recharge_pipelines_ham,
      merged.recharge_sewage_ff_ham, merged.total_recharge_ham,
      merged.stream_recharge_ham, merged.annual_recharge_ham,
      merged.environmental_flows_ham, merged.annual_extractable_gw_ham,
      merged.extraction_domestic_ham, merged.extraction_industrial_ham,
      merged.extraction_irrigation_ham, merged.total_extraction_ham,
      merged.extraction_rate_pct,
      merged.net_availability_future_ham,
      merged.storage_unconfined_fresh_ham, merged.availability_unconfined_fresh,
      merged.availability_confined_fresh, merged.availability_semiconfined_fresh,
      merged.total_gw_availability_ham, merged.status,
      id,
    ]
  );
  return result.rows[0];
}

// ─────────────────────────────────────────────────────────
// DELETE
// ─────────────────────────────────────────────────────────

async function deleteOne(id) {
  const result = await query(
    "DELETE FROM groundwater_data WHERE id = $1 RETURNING id",
    [id]
  );
  return result.rows[0] || null;
}

module.exports = {
  findAll,
  findById,
  insertOne,
  bulkInsert,
  updateOne,
  deleteOne,
  aggregateByState,
  getHotspots,
  getStatusDistribution,
};
