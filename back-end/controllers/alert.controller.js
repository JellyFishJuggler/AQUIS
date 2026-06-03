const { query } = require("../db/pool");

// GET /alerts?limit=20
// Returns districts in CRITICAL or OVER_EXPLOITED status, sorted by extraction rate desc
async function getAlerts(req, res, next) {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 20, 200);

    const result = await query(
      `
      SELECT
        id,
        state,
        district,
        extraction_rate_pct,
        total_extraction_ham,
        annual_extractable_gw_ham,
        rainfall_mm,
        status
      FROM groundwater_data
      WHERE status IN ('CRITICAL', 'OVER_EXPLOITED')
      ORDER BY extraction_rate_pct DESC NULLS LAST
      LIMIT $1
      `,
      [limit]
    );

    res.json({
      meta: { count: result.rows.length },
      alerts: result.rows,
    });
  } catch (err) {
    next(err);
  }
}

// GET /alerts/summary  — count by severity
async function alertSummary(req, res, next) {
  try {
    const result = await query(`
      SELECT
        status,
        COUNT(*) AS count
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
  } catch (err) {
    next(err);
  }
}

module.exports = { getAlerts, alertSummary };
