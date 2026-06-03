/**
 * groundwaterClassification.js
 *
 * Encapsulates AQUIS stress classification logic.
 * Based on CGWB DWLR 2023 methodology (research paper §6.2).
 *
 * Extraction Rate (%) = (Total Extraction / Annual Extractable GW Resource) × 100
 * ─────────────────────────────────────────────────────────────────────────────
 * SAFE          < 70%
 * SEMI_CRITICAL 70–90%
 * CRITICAL      90–100%
 * OVER_EXPLOITED > 100%
 */

const THRESHOLDS = Object.freeze({
  SAFE: 70,
  SEMI_CRITICAL: 90,
  CRITICAL: 100,
});

/**
 * Derive extraction rate from raw values.
 * Returns null if inputs are missing / zero to avoid division-by-zero.
 *
 * @param {number|null} totalExtraction  – total extraction in ha.m
 * @param {number|null} annualExtractable – annual extractable GW resource in ha.m
 * @returns {number|null}
 */
function computeExtractionRate(totalExtraction, annualExtractable) {
  if (
    totalExtraction == null ||
    annualExtractable == null ||
    annualExtractable <= 0
  ) {
    return null;
  }
  const rate = (totalExtraction / annualExtractable) * 100;
  // Cap at 300% (outlier filter from paper §6.1)
  return Math.min(rate, 300);
}

/**
 * Classify a district based on its extraction rate.
 *
 * @param {number|null} extractionRatePct
 * @returns {'SAFE'|'SEMI_CRITICAL'|'CRITICAL'|'OVER_EXPLOITED'|null}
 */
function classifyStatus(extractionRatePct) {
  if (extractionRatePct == null) return null;
  if (extractionRatePct < THRESHOLDS.SAFE) return "SAFE";
  if (extractionRatePct < THRESHOLDS.SEMI_CRITICAL) return "SEMI_CRITICAL";
  if (extractionRatePct <= THRESHOLDS.CRITICAL) return "CRITICAL";
  return "OVER_EXPLOITED";
}

/**
 * Convenience: compute rate AND classify in one call.
 *
 * @param {number|null} totalExtraction
 * @param {number|null} annualExtractable
 * @returns {{ extractionRatePct: number|null, status: string|null }}
 */
function deriveClassification(totalExtraction, annualExtractable) {
  const rate = computeExtractionRate(totalExtraction, annualExtractable);
  const status = classifyStatus(rate);
  return { extractionRatePct: rate, status };
}

/**
 * Validate that submitted extraction rate or resource values are physically plausible.
 *
 * @param {object} fields
 * @returns {{ valid: boolean, errors: string[] }}
 */
function validateExtractionFields(fields) {
  const errors = [];
  const { total_extraction_ham, annual_extractable_gw_ham, extraction_rate_pct } = fields;

  if (
    total_extraction_ham != null &&
    annual_extractable_gw_ham != null &&
    annual_extractable_gw_ham < 0
  ) {
    errors.push("annual_extractable_gw_ham must be non-negative.");
  }

  if (total_extraction_ham != null && total_extraction_ham < 0) {
    errors.push("total_extraction_ham must be non-negative.");
  }

  if (extraction_rate_pct != null && extraction_rate_pct < 0) {
    errors.push("extraction_rate_pct must be non-negative.");
  }

  return { valid: errors.length === 0, errors };
}

module.exports = {
  computeExtractionRate,
  classifyStatus,
  deriveClassification,
  validateExtractionFields,
  THRESHOLDS,
};
