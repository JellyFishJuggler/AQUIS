const { query } = require("../db/pool");

async function getAlerts(req, res, next) {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 20, 200);
    const { state, source } = req.query;

    let sql = `
      SELECT
        id, state, district,
        extraction_rate_pct, total_extraction_ham,
        annual_extractable_gw_ham, rainfall_mm, status
      FROM groundwater_data
      WHERE status IN ('CRITICAL', 'OVER_EXPLOITED')
    `;
    const values = [];
    let idx = 1;

    if (state) {
      sql += ` AND LOWER(state) = LOWER($${idx++})`;
      values.push(state);
    }

    sql += ` ORDER BY extraction_rate_pct DESC NULLS LAST LIMIT $${idx}`;
    values.push(limit);

    const result = await query(sql, values);
    res.json({ meta: { count: result.rows.length }, alerts: result.rows });
  } catch (err) { next(err); }
}

async function alertSummary(req, res, next) {
  try {
    const result = await query(`
      SELECT status, COUNT(*) AS count
      FROM groundwater_data
      WHERE status IN ('CRITICAL', 'OVER_EXPLOITED', 'SEMI_CRITICAL')
      GROUP BY status
      ORDER BY
        CASE status
          WHEN 'OVER_EXPLOITED' THEN 1
          WHEN 'CRITICAL'       THEN 2
          WHEN 'SEMI_CRITICAL'  THEN 3
        END
    `);
    res.json({ data: result.rows });
  } catch (err) { next(err); }
}

module.exports = { getAlerts, alertSummary };
