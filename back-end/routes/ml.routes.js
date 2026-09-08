const router = require("express").Router();
const ctrl = require("../controllers/ml.controller");
const { mlModels, mlModelCompare, mlModelByName, mlForecast, mlAnomalyByStation, mlRiskUnit } = require("../middleware/validators");

router.get("/health",                                       ctrl.getMLHealth);
router.get("/models",             mlModels,                 ctrl.getModels);
router.get("/models/compare",     mlModelCompare,           ctrl.getModelComparison);
router.get("/models/:modelName",  mlModelByName,            ctrl.getModelByName);
router.get("/models/:modelName/metrics", mlModelByName,     ctrl.getModelMetrics);

router.get("/forecast/:stationId", mlForecast,              ctrl.getForecast);
router.get("/anomalies",                                  ctrl.getAnomalies);
router.get("/anomalies/summary",                          ctrl.getAnomalySummary);
router.get("/anomalies/:stationId", mlAnomalyByStation,    ctrl.getAnomalies);

router.get("/risk/summary",                               ctrl.getRiskSummary);
router.get("/risk/priority-areas",                        ctrl.getPriorityAreas);
router.get("/risk/:unitId",      mlRiskUnit,               ctrl.getRisk);

// Live endpoints — headless Python ML service (recency-sorted, slug-keyed)
router.get("/stations",                           ctrl.getLiveStations);
router.get("/stations/:slug",                     ctrl.getLiveStation);
router.get("/districts",                          ctrl.getLiveDistricts);
router.get("/live/models",                        ctrl.getLiveModels);
router.get("/live/forecast/:slug",                ctrl.getLiveForecast);
router.get("/live/fleet/forecasts",               ctrl.getLiveFleetForecasts);
router.get("/live/fleet/recovery",                ctrl.getLiveFleetRecovery);
router.get("/live/fleet/scan",                    ctrl.getLiveFleetScan);
router.post("/assistant/chat",                    ctrl.postAssistantChat);

module.exports = router;
