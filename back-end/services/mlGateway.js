const http = require("http");

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:5000";
const ML_TIMEOUT_MS = parseInt(process.env.ML_TIMEOUT_MS || "60000", 10);

function makeRequest(path, method = "POST", body = null, timeout = ML_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, ML_SERVICE_URL);
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname + url.search,
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

// ---------------------------------------------------------------------------
// Live endpoints backed by the headless Python service (ml/app.py)
//
// These proxies the recency-sorted station/district list, per-station facts,
// trained-model index, XGBoost forecast series, fleet snapshot + recovery rank,
// and the station-locked LLM Data Assistant. All errors surface as JSON.
// ---------------------------------------------------------------------------

function _qs(params = {}) {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") qs.append(k, v);
  }
  const s = qs.toString();
  return s ? `?${s}` : "";
}

async function getStations(params = {}) {
  try {
    const res = await makeRequest(`/stations${_qs(params)}`, "GET");
    return { success: true, data: res.data };
  } catch (err) { return { success: false, error: err.message }; }
}

async function getStation(slug) {
  try {
    const res = await makeRequest(`/stations/${encodeURIComponent(slug)}`, "GET");
    return { success: true, data: res.data };
  } catch (err) { return { success: false, error: err.message }; }
}

async function getDistricts() {
  try {
    const res = await makeRequest(`/districts`, "GET");
    return { success: true, data: res.data };
  } catch (err) { return { success: false, error: err.message }; }
}

async function getLiveForecast(slug, days) {
  try {
    const res = await makeRequest(`/forecast/${encodeURIComponent(slug)}${_qs({ days })}`, "GET");
    return { success: true, data: res.data };
  } catch (err) { return { success: false, error: err.message }; }
}

async function getFleetForecasts() {
  try {
    const res = await makeRequest(`/fleet/forecasts`, "GET");
    return { success: true, data: res.data };
  } catch (err) { return { success: false, error: err.message }; }
}

async function getFleetRecovery(params = {}) {
  try {
    const res = await makeRequest(`/fleet/recovery${_qs(params)}`, "GET");
    return { success: true, data: res.data };
  } catch (err) { return { success: false, error: err.message }; }
}

async function getFleetScan(params = {}) {
  try {
    const res = await makeRequest(`/fleet/scan${_qs(params)}`, "GET");
    return { success: true, data: res.data };
  } catch (err) { return { success: false, error: err.message }; }
}

async function getLiveModels() {
  try {
    const res = await makeRequest(`/models`, "GET");
    return { success: true, data: res.data };
  } catch (err) { return { success: false, error: err.message }; }
}

async function assistantChat(body) {
  try {
    const res = await makeRequest(`/assistant/chat`, "POST", body);
    return { success: true, data: res.data };
  } catch (err) { return { success: false, error: err.message }; }
}

module.exports = {
  healthCheck, getForecast, getAnomalies, getRisk,
  trainModel, getModelComparison, makeRequest,
  getStations, getStation, getDistricts, getLiveForecast,
  getFleetForecasts, getFleetRecovery, getFleetScan, getLiveModels,
  assistantChat,
};
