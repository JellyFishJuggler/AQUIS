const router = require("express").Router();
const ctrl = require("../controllers/ingestion.controller");

router.get("/", ctrl.getRuns);
router.get("/:id", ctrl.getRunById);
router.post("/telemetry", ctrl.triggerTelemetryIngestion);
router.post("/assessment", ctrl.triggerAssessmentIngestion);

module.exports = router;
