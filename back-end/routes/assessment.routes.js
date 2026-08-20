const router = require("express").Router();
const ctrl = require("../controllers/assessment.controller");

router.get("/", ctrl.getMany);
router.get("/summary", ctrl.getSummary);
router.get("/years", ctrl.getAvailableYears);
router.get("/:id", ctrl.getById);
router.get("/:unitId/history", ctrl.getUnitHistory);

module.exports = router;
