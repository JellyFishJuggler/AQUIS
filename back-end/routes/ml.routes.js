const router = require("express").Router();
const ctrl = require("../controllers/ml.controller");

router.get("/health", ctrl.getMLHealth);
router.get("/models", ctrl.getModels);
router.get("/models/compare", ctrl.getModelComparison);
router.get("/models/:modelName", ctrl.getModelByName);
router.get("/models/:modelName/metrics", ctrl.getModelMetrics);

router.get("/forecast/:stationId", ctrl.getForecast);
router.get("/anomalies", ctrl.getAnomalies);
router.get("/anomalies/summary", ctrl.getAnomalySummary);
router.get("/anomalies/:stationId", ctrl.getAnomalies);

router.get("/risk/summary", ctrl.getRiskSummary);
router.get("/risk/priority-areas", ctrl.getPriorityAreas);
router.get("/risk/:unitId", ctrl.getRisk);

module.exports = router;
