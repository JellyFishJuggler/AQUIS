const router = require("express").Router();
const ctrl = require("../controllers/telemetry.controller");
const { telemetryList, telemetryLatest, telemetryByStation } = require("../middleware/validators");

router.get("/",         telemetryList,       ctrl.getMany);
router.get("/latest",   telemetryLatest,     ctrl.getLatest);
router.get("/summary",                       ctrl.getSummary);
router.get("/:stationId", telemetryByStation, ctrl.getByStation);

module.exports = router;
