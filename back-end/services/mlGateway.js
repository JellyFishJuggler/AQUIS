const http = require("http");

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:5000";
const ML_TIMEOUT_MS = parseInt(process.env.ML_TIMEOUT_MS || "60000", 10);

function makeRequest(path, method = "POST", body = null, timeout = ML_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, ML_SERVICE_URL);
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method,
      headers: { "Content-Type": "application/json" },
      timeout,
    };

    const req = http.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => { data += chunk; });
      res.on("end", () => {
        try {
          const parsed = JSON.parse(data);
          resolve({ status: res.statusCode, data: parsed });
        } catch (e) {
          reject(new Error(`Malformed ML response: ${data.slice(0, 200)}`));
        }
      });
    });

    req.on("timeout", () => {
      req.destroy();
      reject(new Error("ML service timeout"));
    });

    req.on("error", (err) => {
      if (err.code === "ECONNREFUSED") {
        reject(new Error("ML service unavailable"));
      } else {
        reject(new Error(`ML service error: ${err.message}`));
      }
    });

    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

async function healthCheck() {
  try {
    const res = await makeRequest("/health", "GET");
    return { available: true, status: res.data };
  } catch (err) {
    return { available: false, error: err.message };
  }
}

async function getForecast(stationId, options = {}) {
  try {
    const res = await makeRequest(`/forecast/${stationId}`, "POST", options);
    return { success: true, data: res.data };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function getAnomalies(stationId, options = {}) {
  try {
    const res = await makeRequest(`/anomalies/${stationId}`, "POST", options);
    return { success: true, data: res.data };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function getRisk(unitId, options = {}) {
  try {
    const res = await makeRequest(`/risk/${unitId}`, "POST", options);
    return { success: true, data: res.data };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function trainModel(task, options = {}) {
  try {
    const res = await makeRequest(`/train`, "POST", { task, ...options });
    return { success: true, data: res.data };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function getModelComparison(task) {
  try {
    const res = await makeRequest(`/models/compare`, "POST", { task });
    return { success: true, data: res.data };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

module.exports = {
  healthCheck, getForecast, getAnomalies, getRisk,
  trainModel, getModelComparison, makeRequest,
};
