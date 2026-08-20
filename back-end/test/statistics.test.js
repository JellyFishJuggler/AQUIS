const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const { mankendall, sensSlope } = require("../services/statisticsService");

describe("statisticsService", () => {
  describe("mankendall", () => {
    it("detects increasing trend with large enough sample", () => {
      const data = [];
      for (let i = 0; i < 30; i++) {
        data.push({ time: i, value: 10 + i * 2 + (Math.sin(i) * 0.5) });
      }
      const result = mankendall(data);
      assert.ok(result);
      assert.equal(result.direction, "increasing");
      assert.equal(result.significant, true);
      assert.ok(result.s_statistic > 0);
    });

    it("detects decreasing trend with large enough sample", () => {
      const data = [];
      for (let i = 0; i < 30; i++) {
        data.push({ time: i, value: 100 - i * 2 + (Math.sin(i) * 0.5) });
      }
      const result = mankendall(data);
      assert.ok(result);
      assert.equal(result.direction, "decreasing");
      assert.equal(result.significant, true);
      assert.ok(result.s_statistic < 0);
    });

    it("returns no trend for random data", () => {
      const data = [];
      const values = [15, 13, 17, 14, 16, 12, 18, 11, 19, 10, 20, 9, 14, 16, 13, 17, 15, 12, 18, 11];
      values.forEach((v, i) => data.push({ time: i, value: v }));
      const result = mankendall(data);
      assert.ok(result);
      assert.equal(result.direction, "no trend");
      assert.equal(result.significant, false);
    });

    it("returns null for insufficient data", () => {
      const result = mankendall([{ time: 1, value: 10 }, { time: 2, value: 20 }]);
      assert.equal(result, null);
    });

    it("returns correct n count", () => {
      const data = Array.from({ length: 10 }, (_, i) => ({ time: i, value: i }));
      const result = mankendall(data);
      assert.ok(result);
      assert.equal(result.n, 10);
    });
  });

  describe("sensSlope", () => {
    it("computes positive slope for increasing data", () => {
      const data = [
        { time: 1, value: 10 },
        { time: 2, value: 12 },
        { time: 3, value: 14 },
        { time: 4, value: 16 },
      ];
      const result = sensSlope(data);
      assert.equal(result.slope > 0, true);
      assert.ok(Math.abs(result.slope - 2) < 0.01);
    });

    it("computes negative slope for decreasing data", () => {
      const data = [
        { time: 1, value: 20 },
        { time: 2, value: 15 },
        { time: 3, value: 10 },
        { time: 4, value: 5 },
      ];
      const result = sensSlope(data);
      assert.ok(result.slope < 0);
    });

    it("returns null for insufficient data", () => {
      assert.equal(sensSlope([{ time: 1, value: 10 }]), null);
    });

    it("returns intercept and slope_per_year", () => {
      const data = Array.from({ length: 10 }, (_, i) => ({ time: i * 86400, value: i * 10 }));
      const result = sensSlope(data);
      assert.ok(result);
      assert.ok(typeof result.intercept === "number");
      assert.ok(typeof result.slope_per_year === "number");
    });
  });
});
