const { validationResult, query: vQuery, param, body } = require("express-validator");

/**
 * Run express-validator result check; if errors exist, respond 400.
 */
function validate(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  next();
}

// ── Reusable validator chains ─────────────────────────────

const paginationRules = [
  vQuery("limit")
    .optional()
    .isInt({ min: 1, max: 500 })
    .withMessage("limit must be 1–500")
    .toInt(),
  vQuery("offset")
    .optional()
    .isInt({ min: 0 })
    .withMessage("offset must be ≥ 0")
    .toInt(),
];

const sortRules = [
  vQuery("sort_by")
    .optional()
    .isIn(["id", "state", "district", "extraction_rate_pct", "total_extraction_ham", "rainfall_mm", "status"])
    .withMessage("Invalid sort_by field"),
  vQuery("order")
    .optional()
    .isIn(["asc", "desc"])
    .withMessage("order must be 'asc' or 'desc'"),
];

const idRule = [
  param("id").isInt({ min: 1 }).withMessage("id must be a positive integer").toInt(),
];

const dataBodyRules = [
  body("state").notEmpty().withMessage("state is required"),
  body("district").notEmpty().withMessage("district is required"),
  body("rainfall_mm").optional({ nullable: true }).isFloat({ min: 0 }),
  body("total_extraction_ham").optional({ nullable: true }).isFloat({ min: 0 }),
  body("annual_extractable_gw_ham").optional({ nullable: true }).isFloat({ min: 0 }),
];

module.exports = {
  validate,
  paginationRules,
  sortRules,
  idRule,
  dataBodyRules,
};
