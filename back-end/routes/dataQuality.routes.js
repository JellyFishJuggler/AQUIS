const router = require("express").Router();
const ctrl = require("../controllers/dataQuality.controller");

router.get("/telemetry", ctrl.getTelemetrySummary);
router.get("/assessment", ctrl.getAssessmentQuality);
router.get("/assessment/:year", ctrl.getAssessmentQuality);
router.get("/:stationId", ctrl.getStationQuality);

module.exports = router;
