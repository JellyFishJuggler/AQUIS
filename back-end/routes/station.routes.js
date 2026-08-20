const router = require("express").Router();
const ctrl = require("../controllers/station.controller");
const { stationList, stationNearby, stationById } = require("../middleware/validators");

router.get("/",          stationList,      ctrl.getAll);
router.get("/nearby",    stationNearby,    ctrl.getNearby);
router.get("/state-summary",               ctrl.getStateSummary);
router.get("/district-summary",            ctrl.getDistrictSummary);
router.get("/:stationId", stationById,     ctrl.getById);

module.exports = router;
