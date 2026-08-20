const { validationResult, query: vQuery, param, body } = require("express-validator");

function validate(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  next();
}

// ── Reusable chains ────────────────────────────────────────

const paginationRules = [
  vQuery("limit")
    .optional()
    .isInt({ min: 1, max: 5000 })
    .withMessage("limit must be 1–5000")
    .toInt(),
  vQuery("offset")
    .optional()
    .isInt({ min: 0 })
    .withMessage("offset must be >= 0")
    .toInt(),
];

const smallPaginationRules = [
  vQuery("limit")
    .optional()
    .isInt({ min: 1, max: 500 })
    .withMessage("limit must be 1–500")
    .toInt(),
  vQuery("offset")
    .optional()
    .isInt({ min: 0 })
    .withMessage("offset must be >= 0")
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

const stationIdParam = [
  param("stationId").isInt({ min: 1 }).withMessage("stationId must be a positive integer").toInt(),
];

const unitIdParam = [
  param("unitId").isInt({ min: 1 }).withMessage("unitId must be a positive integer").toInt(),
];

const stationIdQuery = [
  vQuery("stationId")
    .optional()
    .isInt({ min: 1 })
    .withMessage("stationId must be a positive integer")
    .toInt(),
];

const dateRangeRules = [
  vQuery("startDate")
    .optional()
    .isISO8601()
    .withMessage("startDate must be a valid ISO 8601 date"),
  vQuery("endDate")
    .optional()
    .isISO8601()
    .withMessage("endDate must be a valid ISO 8601 date"),
];

const textFilterRules = [
  vQuery("state")
    .optional()
    .isLength({ min: 1, max: 100 })
    .withMessage("state must be 1–100 characters")
    .trim()
    .escape(),
  vQuery("district")
    .optional()
    .isLength({ min: 1, max: 100 })
    .withMessage("district must be 1–100 characters")
    .trim()
    .escape(),
  vQuery("agency")
    .optional()
    .isLength({ min: 1, max: 100 })
    .withMessage("agency must be 1–100 characters")
    .trim()
    .escape(),
];

const yearRule = [
  vQuery("year")
    .optional()
    .matches(/^\d{4}-\d{4}$/)
    .withMessage("year must be in format YYYY-YYYY (e.g. 2024-2025)"),
];

const assessmentYearRule = [
  param("year")
    .matches(/^\d{4}-\d{4}$/)
    .withMessage("year must be in format YYYY-YYYY"),
];

const statusRule = [
  vQuery("status")
    .optional()
    .isIn(["SAFE", "SEMI_CRITICAL", "CRITICAL", "OVER_EXPLOITED"])
    .withMessage("status must be SAFE, SEMI_CRITICAL, CRITICAL, or OVER_EXPLOITED"),
];

const taskRule = [
  vQuery("task")
    .optional()
    .isIn(["forecast", "anomaly", "risk"])
    .withMessage("task must be forecast, anomaly, or risk"),
];

const modelNameRule = [
  param("modelName")
    .isLength({ min: 1, max: 100 })
    .withMessage("modelName is required")
    .trim()
    .matches(/^[a-zA-Z0-9_-]+$/)
    .withMessage("modelName must contain only letters, numbers, hyphens, underscores"),
];

const nearbyRules = [
  vQuery("lat")
    .exists()
    .withMessage("lat is required")
    .isFloat({ min: -90, max: 90 })
    .withMessage("lat must be between -90 and 90")
    .toFloat(),
  vQuery("lon")
    .exists()
    .withMessage("lon is required")
    .isFloat({ min: -180, max: 180 })
    .withMessage("lon must be between -180 and 180")
    .toFloat(),
  vQuery("limit")
    .optional()
    .isInt({ min: 1, max: 100 })
    .withMessage("limit must be 1–100")
    .toInt(),
];

const ingestionBodyRules = [
  body("batchSize")
    .optional()
    .isInt({ min: 1, max: 1000 })
    .withMessage("batchSize must be 1–1000")
    .toInt(),
  body("limit")
    .optional()
    .isInt({ min: 1, max: 100000 })
    .withMessage("limit must be 1–100000")
    .toInt(),
];

const assessmentIngestionBodyRules = [
  body("assessmentYear")
    .optional()
    .matches(/^\d{4}-\d{4}$/)
    .withMessage("assessmentYear must be in format YYYY-YYYY"),
  body("sourceDir")
    .optional()
    .isLength({ min: 1, max: 500 })
    .withMessage("sourceDir path too long"),
];

// ── Route-specific validator groups ────────────────────────

const stationList = [...paginationRules, ...textFilterRules, validate];
const stationNearby = [...nearbyRules, validate];
const stationById = [...stationIdParam, validate];

const telemetryList = [...paginationRules, ...textFilterRules, ...stationIdQuery, ...dateRangeRules, validate];
const telemetryLatest = [
  vQuery("stationId").exists().withMessage("stationId is required").isInt({ min: 1 }).toInt(),
  validate,
];
const telemetryByStation = [...stationIdParam, ...dateRangeRules, ...paginationRules, validate];

const assessmentList = [...paginationRules, ...textFilterRules, ...yearRule, ...statusRule,
  vQuery("unitId").optional().isInt({ min: 1 }).toInt(),
  validate,
];
const assessmentById = [...idRule, validate];
const assessmentHistory = [...unitIdParam, validate];

const trendByStation = [...stationIdParam, validate];

const mlForecast = [...stationIdParam, validate];
const mlAnomalyByStation = [...stationIdParam, ...smallPaginationRules, validate];
const mlRiskUnit = [...unitIdParam, validate];

const mlModels = [...smallPaginationRules, ...taskRule, validate];
const mlModelCompare = [
  vQuery("task").exists().withMessage("task is required").isIn(["forecast", "anomaly", "risk"]),
  validate,
];
const mlModelByName = [...modelNameRule, validate];

const mlDataTelemetry = [...paginationRules, ...textFilterRules, ...stationIdQuery, ...dateRangeRules, validate];
const mlDataTelemetryStation = [...stationIdParam, ...dateRangeRules, ...paginationRules, validate];
const mlDataAssessment = [...paginationRules, ...textFilterRules, ...yearRule,
  vQuery("unitId").optional().isInt({ min: 1 }).toInt(),
  validate,
];
const mlDataAssessmentUnit = [...unitIdParam, validate];

const dataQualityStation = [...stationIdParam, validate];
const dataQualityAssessmentYear = [...assessmentYearRule, validate];

const ingestionRuns = [...smallPaginationRules,
  vQuery("source").optional().isIn(["telemetry_api", "assessment_excel"]).withMessage("invalid source"),
  vQuery("status").optional().isIn(["running", "completed", "failed", "partial"]),
  validate,
];
const ingestionById = [...idRule, validate];

// ── Legacy ─────────────────────────────────────────────────

const dataBodyRules = [
  body("state").notEmpty().withMessage("state is required"),
  body("district").notEmpty().withMessage("district is required"),
  body("rainfall_mm").optional({ nullable: true }).isFloat({ min: 0 }),
  body("total_extraction_ham").optional({ nullable: true }).isFloat({ min: 0 }),
  body("annual_extractable_gw_ham").optional({ nullable: true }).isFloat({ min: 0 }),
];

module.exports = {
  validate,

  stationList, stationNearby, stationById,
  telemetryList, telemetryLatest, telemetryByStation,
  assessmentList, assessmentById, assessmentHistory,
  trendByStation,
  mlModels, mlModelCompare, mlModelByName, mlForecast, mlAnomalyByStation, mlRiskUnit,
  mlDataTelemetry, mlDataTelemetryStation,
  mlDataAssessment, mlDataAssessmentUnit,
  dataQualityStation, dataQualityAssessmentYear,
  ingestionRuns, ingestionById,

  stationIdParam, unitIdParam,
  paginationRules, sortRules, idRule, dataBodyRules,
};
