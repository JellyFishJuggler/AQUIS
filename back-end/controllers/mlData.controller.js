const { query } = require("../db/pool");
const telemetryService = require("../services/telemetryService");
const assessmentService = require("../services/assessmentService");

async function getTelemetryData(req, res, next) {
  try {
    const { state, district, stationId, startDate, endDate, limit = 500, offset = 0 } = req.query;
    const result = await telemetryService.findMany({
      state, district, stationId: stationId ? parseInt(stationId) : undefined,
      startDate, endDate, limit: parseInt(limit), offset: parseInt(offset),
    });
    res.json({
      meta: { total: result.total, limit: parseInt(limit), offset: parseInt(offset), type: "telemetry" },
      data: result.rows.map(r => ({
        station_id: r.station_id,
        timestamp: r.observed_at,
        groundwater_level: r.groundwater_level,
        state: r.state,
        district: r.district,
        latitude: r.latitude,
        longitude: r.longitude,
        station_name: r.station_name,
      })),
    });
  } catch (err) { next(err); }
}

async function getTelemetryByStation(req, res, next) {
  try {
    const { stationId } = req.params;
    const { startDate, endDate, limit = 5000, offset = 0 } = req.query;
    const result = await telemetryService.findByStationId(parseInt(stationId), {
      startDate, endDate, limit: parseInt(limit), offset: parseInt(offset),
    });
    const station = require("../services/stationService").findById(parseInt(stationId));
    const st = await station;
    res.json({
      meta: { total: result.total, limit: parseInt(limit), offset: parseInt(offset), type: "telemetry_station" },
      station: st ? { id: st.id, name: st.station_name, state: st.state, district: st.district, lat: st.latitude, lon: st.longitude } : null,
      data: result.rows.map(r => ({
        station_id: r.station_id,
        timestamp: r.observed_at,
        groundwater_level: r.groundwater_level,
      })),
    });
  } catch (err) { next(err); }
}

async function getAssessmentData(req, res, next) {
  try {
    const { state, district, year, unitId, limit = 500, offset = 0 } = req.query;
    const result = await assessmentService.findRecords({
      state, district, year,
      unitId: unitId ? parseInt(unitId) : undefined,
      limit: parseInt(limit), offset: parseInt(offset),
    });
    res.json({
      meta: { total: result.total, limit: parseInt(limit), offset: parseInt(offset), type: "assessment" },
      data: result.rows.map(r => ({
        assessment_unit_id: r.assessment_unit_id,
        assessment_year: r.assessment_year,
        state: r.state,
        district: r.district,
        assessment_unit: r.assessment_unit,
        recharge: r.recharge_rainfall_ham,
        total_extraction: r.total_extraction_ham,
        annual_extractable_resource: r.annual_extractable_gw_ham,
        stage_of_extraction: r.stage_of_extraction_pct,
        extraction_rate_pct: r.extraction_rate_pct,
        resource_category: r.status,
        annual_recharge: r.annual_recharge_ham,
        rainfall_mm: r.rainfall_mm,
      })),
    });
  } catch (err) { next(err); }
}

async function getAssessmentByUnit(req, res, next) {
  try {
    const { unitId } = req.params;
    const history = await assessmentService.getUnitHistory(parseInt(unitId));
    res.json({
      meta: { total: history.length, type: "assessment_unit_history" },
      data: history.map(r => ({
        assessment_year: r.assessment_year,
        recharge: r.recharge_rainfall_ham,
        total_extraction: r.total_extraction_ham,
        annual_extractable_resource: r.annual_extractable_gw_ham,
        stage_of_extraction: r.stage_of_extraction_pct,
        extraction_rate_pct: r.extraction_rate_pct,
        resource_category: r.status,
      })),
    });
  } catch (err) { next(err); }
}

module.exports = { getTelemetryData, getTelemetryByStation, getAssessmentData, getAssessmentByUnit };
