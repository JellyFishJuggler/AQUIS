const router = require("express").Router();
const ctrl = require("../controllers/mlData.controller");

router.get("/telemetry", ctrl.getTelemetryData);
router.get("/telemetry/:stationId", ctrl.getTelemetryByStation);
router.get("/assessment", ctrl.getAssessmentData);
router.get("/assessment/:unitId", ctrl.getAssessmentByUnit);

module.exports = router;
