const router = require("express").Router();
const ctrl = require("../controllers/telemetry.controller");

router.get("/", ctrl.getMany);
router.get("/latest", ctrl.getLatest);
router.get("/summary", ctrl.getSummary);
router.get("/:stationId", ctrl.getByStation);

module.exports = router;
