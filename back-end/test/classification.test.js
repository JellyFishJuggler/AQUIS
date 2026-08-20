const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const {
  computeExtractionRate,
  classifyStatus,
  deriveClassification,
  validateExtractionFields,
} = require("../services/groundwaterClassification");

describe("groundwaterClassification", () => {
  describe("computeExtractionRate", () => {
    it("returns correct percentage", () => {
      assert.equal(computeExtractionRate(700, 1000), 70);
    });
    it("returns null for missing values", () => {
      assert.equal(computeExtractionRate(null, 1000), null);
      assert.equal(computeExtractionRate(700, null), null);
    });
    it("returns null for zero denominator", () => {
      assert.equal(computeExtractionRate(700, 0), null);
    });
    it("caps at 300%", () => {
      assert.equal(computeExtractionRate(5000, 1000), 300);
    });
  });

  describe("classifyStatus", () => {
    it("classifies SAFE", () => assert.equal(classifyStatus(50), "SAFE"));
    it("classifies SEMI_CRITICAL", () => assert.equal(classifyStatus(80), "SEMI_CRITICAL"));
    it("classifies CRITICAL", () => assert.equal(classifyStatus(95), "CRITICAL"));
    it("classifies OVER_EXPLOITED", () => assert.equal(classifyStatus(150), "OVER_EXPLOITED"));
    it("returns null for null input", () => assert.equal(classifyStatus(null), null));
    it("boundary: exactly 70 is SEMI_CRITICAL", () => assert.equal(classifyStatus(70), "SEMI_CRITICAL"));
    it("boundary: exactly 90 is CRITICAL", () => assert.equal(classifyStatus(90), "CRITICAL"));
    it("boundary: exactly 100 is CRITICAL (OVER is >100)", () => assert.equal(classifyStatus(100), "CRITICAL"));
    it("boundary: 100.01 is OVER_EXPLOITED", () => assert.equal(classifyStatus(100.01), "OVER_EXPLOITED"));
  });

  describe("deriveClassification", () => {
    it("returns both rate and status", () => {
      const result = deriveClassification(800, 1000);
      assert.equal(result.extractionRatePct, 80);
      assert.equal(result.status, "SEMI_CRITICAL");
    });
  });

  describe("validateExtractionFields", () => {
    it("passes for valid fields", () => {
      const result = validateExtractionFields({ total_extraction_ham: 100, annual_extractable_gw_ham: 200 });
      assert.equal(result.valid, true);
    });
    it("fails for negative extraction", () => {
      const result = validateExtractionFields({ total_extraction_ham: -100 });
      assert.equal(result.valid, false);
    });
  });
});
