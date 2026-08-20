const router = require("express").Router();
const ctrl = require("../controllers/station.controller");

router.get("/", ctrl.getAll);
router.get("/nearby", ctrl.getNearby);
router.get("/state-summary", ctrl.getStateSummary);
router.get("/district-summary", ctrl.getDistrictSummary);
router.get("/:stationId", ctrl.getById);

module.exports = router;
