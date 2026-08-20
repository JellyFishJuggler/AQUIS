const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const { parseTimestamp, parseNumeric, cleanText } = require("../services/telemetryIngestion");

describe("telemetryIngestion utilities", () => {
  describe("parseTimestamp", () => {
    it("parses DD-MM-YYYY HH:MM format", () => {
      const result = parseTimestamp("23-02-2024 18:00");
      assert.ok(result);
      assert.ok(result.includes("2024"));
    });
    it("parses YYYY-MM-DD HH:MM format", () => {
      const result = parseTimestamp("2024-02-23 18:00");
      assert.ok(result);
    });
    it("returns null for dash", () => {
      assert.equal(parseTimestamp("-"), null);
    });
    it("returns null for empty", () => {
      assert.equal(parseTimestamp(""), null);
      assert.equal(parseTimestamp(null), null);
    });
  });

  describe("parseNumeric", () => {
    it("parses valid numbers", () => {
      assert.equal(parseNumeric("3.14"), 3.14);
      assert.equal(parseNumeric(-5.2), -5.2);
    });
    it("returns null for invalid values", () => {
      assert.equal(parseNumeric("-"), null);
      assert.equal(parseNumeric(""), null);
      assert.equal(parseNumeric(null), null);
      assert.equal(parseNumeric("NA"), null);
    });
  });

  describe("cleanText", () => {
    it("trims whitespace", () => {
      assert.equal(cleanText("  hello  "), "hello");
    });
    it("returns null for dash/empty/NA", () => {
      assert.equal(cleanText("-"), null);
      assert.equal(cleanText(""), null);
      assert.equal(cleanText("N/A"), null);
      assert.equal(cleanText(null), null);
    });
  });
});
