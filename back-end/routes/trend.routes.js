const router = require("express").Router();
const ctrl = require("../controllers/trend.controller");

router.get("/summary", ctrl.getTrendSummary);
router.get("/:stationId", ctrl.getStationTrend);

module.exports = router;
