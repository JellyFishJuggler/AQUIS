const router = require("express").Router();
const ctrl = require("../controllers/trend.controller");
const { trendByStation } = require("../middleware/validators");

router.get("/summary",                        ctrl.getTrendSummary);
router.get("/:stationId", trendByStation,     ctrl.getStationTrend);

module.exports = router;
