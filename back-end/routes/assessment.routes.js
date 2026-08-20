const router = require("express").Router();
const ctrl = require("../controllers/assessment.controller");
const { assessmentList, assessmentById, assessmentHistory } = require("../middleware/validators");

router.get("/",          assessmentList,    ctrl.getMany);
router.get("/summary",                      ctrl.getSummary);
router.get("/years",                        ctrl.getAvailableYears);
router.get("/:id",        assessmentById,   ctrl.getById);
router.get("/:unitId/history", assessmentHistory, ctrl.getUnitHistory);

module.exports = router;
