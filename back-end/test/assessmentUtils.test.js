const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const { buildExternalUnitId } = require("../services/assessmentService");

describe("assessmentService utilities", () => {
  describe("buildExternalUnitId", () => {
    it("combines state, district, unit", () => {
      const id = buildExternalUnitId("Bihar", "Arwal", "Arwal");
      assert.ok(id.includes("BIHAR"));
      assert.ok(id.includes("ARWAL"));
    });
    it("handles null unit", () => {
      const id = buildExternalUnitId("Bihar", "Arwal", null);
      assert.ok(id.includes("BIHAR"));
    });
    it("normalizes case", () => {
      const id1 = buildExternalUnitId("bihar", "arwal", "arwal");
      const id2 = buildExternalUnitId("BIHAR", "ARWAL", "ARWAL");
      assert.equal(id1, id2);
    });
  });
});
