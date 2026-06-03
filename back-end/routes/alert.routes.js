const router = require("express").Router();
const ctrl   = require("../controllers/alert.controller");

router.get("/",       ctrl.getAlerts);
router.get("/summary", ctrl.alertSummary);

module.exports = router;
