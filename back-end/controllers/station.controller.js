const stationService = require("../services/stationService");

async function getAll(req, res, next) {
  try {
    const { state, district, agency, limit = 50, offset = 0 } = req.query;
    const result = await stationService.findAll({
      state, district, agency,
      limit: parseInt(limit), offset: parseInt(offset),
    });
    res.json({ meta: { total: result.total, limit: parseInt(limit), offset: parseInt(offset) }, data: result.rows });
  } catch (err) { next(err); }
}

async function getById(req, res, next) {
  try {
    const station = await stationService.findById(req.params.stationId);
    if (!station) return res.status(404).json({ error: "Station not found" });
    res.json(station);
  } catch (err) { next(err); }
}

async function getNearby(req, res, next) {
  try {
    const { lat, lon, limit = 20 } = req.query;
    if (!lat || !lon) return res.status(400).json({ error: "lat and lon are required" });
    const stations = await stationService.findNearby(parseFloat(lat), parseFloat(lon), 50, parseInt(limit));
    res.json({ data: stations });
  } catch (err) { next(err); }
}

async function getStateSummary(req, res, next) {
  try {
    const data = await stationService.getStateSummary();
    res.json({ data });
  } catch (err) { next(err); }
}

async function getDistrictSummary(req, res, next) {
  try {
    const { state } = req.query;
    const data = await stationService.getDistrictSummary(state);
    res.json({ data });
  } catch (err) { next(err); }
}

module.exports = { getAll, getById, getNearby, getStateSummary, getDistrictSummary };
