const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const mlGateway = require("../services/mlGateway");

describe("mlGateway", () => {
  it("healthCheck returns unavailable when service is down", async () => {
    const result = await mlGateway.healthCheck();
    assert.equal(result.available, false);
    assert.ok(result.error);
  });

  it("getForecast handles service unavailable", async () => {
    const result = await mlGateway.getForecast(1);
    assert.equal(result.success, false);
    assert.ok(result.error);
  });

  it("getAnomalies handles service unavailable", async () => {
    const result = await mlGateway.getAnomalies(1);
    assert.equal(result.success, false);
    assert.ok(result.error);
  });

  it("getRisk handles service unavailable", async () => {
    const result = await mlGateway.getRisk(1);
    assert.equal(result.success, false);
    assert.ok(result.error);
  });
});
