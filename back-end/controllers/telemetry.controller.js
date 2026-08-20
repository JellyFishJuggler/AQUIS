const telemetryService = require("../services/telemetryService");

async function getMany(req, res, next) {
  try {
    const { state, district, stationId, startDate, endDate, limit = 100, offset = 0 } = req.query;
    const result = await telemetryService.findMany({
      state, district, stationId: stationId ? parseInt(stationId) : undefined,
      startDate, endDate,
      limit: parseInt(limit), offset: parseInt(offset),
    });
    res.json({ meta: { total: result.total, limit: parseInt(limit), offset: parseInt(offset) }, data: result.rows });
  } catch (err) { next(err); }
}

async function getLatest(req, res, next) {
  try {
    const { stationId } = req.query;
    if (!stationId) {
      return res.status(400).json({ error: "stationId query parameter is required" });
    }
    const obs = await telemetryService.getLatestByStation(parseInt(stationId));
    if (!obs) return res.status(404).json({ error: "No telemetry data found for this station" });
    res.json(obs);
  } catch (err) { next(err); }
}

async function getByStation(req, res, next) {
  try {
    const { stationId } = req.params;
    const { startDate, endDate, limit = 1000, offset = 0 } = req.query;
    const result = await telemetryService.findByStationId(parseInt(stationId), {
      startDate, endDate, limit: parseInt(limit), offset: parseInt(offset),
    });
    res.json({ meta: { total: result.total, limit: parseInt(limit), offset: parseInt(offset) }, data: result.rows });
  } catch (err) { next(err); }
}

async function getSummary(req, res, next) {
  try {
    const data = await telemetryService.getStateDistrictSummary();
    res.json({ data });
  } catch (err) { next(err); }
}

module.exports = { getMany, getLatest, getByStation, getSummary };
