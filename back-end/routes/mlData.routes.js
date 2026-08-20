const router = require("express").Router();
const ctrl = require("../controllers/mlData.controller");
const { mlDataTelemetry, mlDataTelemetryStation, mlDataAssessment, mlDataAssessmentUnit } = require("../middleware/validators");

router.get("/telemetry",              mlDataTelemetry,        ctrl.getTelemetryData);
router.get("/telemetry/:stationId",   mlDataTelemetryStation, ctrl.getTelemetryByStation);
router.get("/assessment",             mlDataAssessment,       ctrl.getAssessmentData);
router.get("/assessment/:unitId",     mlDataAssessmentUnit,   ctrl.getAssessmentByUnit);

module.exports = router;
