const router = require("express").Router();
const ctrl = require("../controllers/dataQuality.controller");
const { dataQualityStation, dataQualityAssessmentYear, validate } = require("../middleware/validators");

router.get("/telemetry",                                  ctrl.getTelemetrySummary);
router.get("/assessment",  dataQualityAssessmentYear,     ctrl.getAssessmentQuality);
router.get("/assessment/:year", dataQualityAssessmentYear, ctrl.getAssessmentQuality);
router.get("/:stationId",  dataQualityStation,            ctrl.getStationQuality);

module.exports = router;
