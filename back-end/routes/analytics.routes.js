const router = require("express").Router();
const ctrl   = require("../controllers/analytics.controller");

router.get("/state-summary",       ctrl.stateSummary);
router.get("/hotspots",            ctrl.hotspots);
router.get("/status-distribution", ctrl.statusDistribution);

module.exports = router;
