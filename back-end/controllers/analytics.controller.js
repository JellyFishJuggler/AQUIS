const repo = require("../services/groundwaterRepository");

async function stateSummary(req, res, next) {
  try {
    const rows = await repo.aggregateByState(req.query.state || null);
    res.json({ data: rows });
  } catch (err) { next(err); }
}

async function hotspots(req, res, next) {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 10, 100);
    const rows = await repo.getHotspots(limit);
    res.json({ data: rows });
  } catch (err) { next(err); }
}

async function statusDistribution(req, res, next) {
  try {
    const rows = await repo.getStatusDistribution();
    res.json({ data: rows });
  } catch (err) { next(err); }
}

module.exports = { stateSummary, hotspots, statusDistribution };
