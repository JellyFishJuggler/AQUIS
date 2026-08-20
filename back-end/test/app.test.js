const { describe, it } = require("node:test");
const assert = require("node:assert/strict");

describe("app configuration", () => {
  it("app exports an express instance", () => {
    const app = require("../app");
    assert.ok(app);
    assert.equal(typeof app, "function");
  });

  it("health endpoint returns ok", async () => {
    const http = require("http");
    const app = require("../app");

    const server = app.listen(0);
    const port = server.address().port;

    try {
      const result = await new Promise((resolve, reject) => {
        http.get(`http://localhost:${port}/health`, (res) => {
          let data = "";
          res.on("data", (chunk) => { data += chunk; });
          res.on("end", () => resolve(JSON.parse(data)));
        }).on("error", reject);
      });
      assert.equal(result.status, "ok");
      assert.ok(result.timestamp);
    } finally {
      server.close();
    }
  });
});
