const dataQualityService = require("../services/dataQualityService");

async function getStationQuality(req, res, next) {
  try {
    const { stationId } = req.params;
    const data = await dataQualityService.getStationQuality(parseInt(stationId));
    res.json({ data });
  } catch (err) { next(err); }
}

async function getTelemetrySummary(req, res, next) {
  try {
    const data = await dataQualityService.getTelemetryQualitySummary();
    res.json({ data });
  } catch (err) { next(err); }
}

async function getAssessmentQuality(req, res, next) {
  try {
    const { year } = req.query;
    const data = await dataQualityService.getAssessmentQualitySummary(year);
    res.json({ data });
  } catch (err) { next(err); }
}

module.exports = { getStationQuality, getTelemetrySummary, getAssessmentQuality };
