const router = require("express").Router();
const ctrl   = require("../controllers/data.controller");
const {
  validate, paginationRules, sortRules, idRule, dataBodyRules,
} = require("../middleware/validators");

router.get(   "/",    [...paginationRules, ...sortRules], validate, ctrl.getAll);
router.get(   "/:id", idRule,                             validate, ctrl.getOne);
router.post(  "/",    dataBodyRules,                      validate, ctrl.createOne);
router.put(   "/:id", [...idRule, ...dataBodyRules],      validate, ctrl.updateOne);
router.delete("/:id", idRule,                             validate, ctrl.deleteOne);

module.exports = router;
