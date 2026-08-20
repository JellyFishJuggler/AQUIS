const statisticsService = require("../services/statisticsService");

async function getStationTrend(req, res, next) {
  try {
    const { stationId } = req.params;
    const trend = await statisticsService.computeStationTrend(parseInt(stationId));
    res.json(trend);
  } catch (err) { next(err); }
}

async function getTrendSummary(req, res, next) {
  try {
    const summary = await statisticsService.computeTrendSummary();
    res.json(summary);
  } catch (err) { next(err); }
}

module.exports = { getStationTrend, getTrendSummary };
