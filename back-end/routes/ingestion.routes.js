const router = require("express").Router();
const ctrl = require("../controllers/ingestion.controller");
const { ingestionRuns, ingestionById, validate } = require("../middleware/validators");

router.get("/",             ingestionRuns,  ctrl.getRuns);
router.get("/:id",          ingestionById,  ctrl.getRunById);
router.post("/telemetry", validate,         ctrl.triggerTelemetryIngestion);
router.post("/assessment", validate,        ctrl.triggerAssessmentIngestion);

module.exports = router;
