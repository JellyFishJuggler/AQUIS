const { query } = require("../db/pool");

function mankendall(data) {
  const n = data.length;
  if (n < 4) return null;

  let s = 0;
  for (let k = 0; k < n - 1; k++) {
    for (let j = k + 1; j < n; j++) {
      const diff = data[j].value - data[k].value;
      if (diff > 0) s += 1;
      else if (diff < 0) s -= 1;
    }
  }

  const uniqueVals = {};
  data.forEach(d => { uniqueVals[d.value] = (uniqueVals[d.value] || 0) + 1; });
  let tp = 0;
  Object.values(uniqueVals).forEach(t => {
    if (t > 1) tp += t * (t - 1) * (2 * t + 5);
  });

  const varS = (n * (n - 1) * (2 * n + 5) - tp) / 18;

  let z;
  if (s > 0) z = (s - 1) / Math.sqrt(varS);
  else if (s < 0) z = (s + 1) / Math.sqrt(varS);
  else z = 0;

  const twoTailedP = 2 * (1 - normalCDF(Math.abs(z)));
  let direction = "no trend";
  if (z > 0 && twoTailedP < 0.05) direction = "increasing";
  else if (z < 0 && twoTailedP < 0.05) direction = "decreasing";

  const significant = twoTailedP < 0.05;

  return {
    s_statistic: s,
    z_score: parseFloat(z.toFixed(6)),
    p_value: parseFloat(twoTailedP.toFixed(6)),
    direction,
    significant,
    n,
    variance: parseFloat(varS.toFixed(4)),
  };
}

function normalCDF(x) {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1.0 + sign * y);
}

function sensSlope(data) {
  const n = data.length;
  if (n < 4) return null;

  const slopes = [];
  for (let k = 0; k < n - 1; k++) {
    for (let j = k + 1; j < n; j++) {
      const dt = data[j].time - data[k].time;
      if (dt !== 0) {
        slopes.push((data[j].value - data[k].value) / dt);
      }
    }
  }

  slopes.sort((a, b) => a - b);
  const median = slopes.length % 2 === 0
    ? (slopes[slopes.length / 2 - 1] + slopes[slopes.length / 2]) / 2
    : slopes[Math.floor(slopes.length / 2)];

  const intercepts = data.map(d => d.value - median * d.time);
  intercepts.sort((a, b) => a - b);
  const interceptMedian = intercepts.length % 2 === 0
    ? (intercepts[intercepts.length / 2 - 1] + intercepts[intercepts.length / 2]) / 2
    : intercepts[Math.floor(intercepts.length / 2)];

  const sorted = [...data].sort((a, b) => a.time - b.time);
  const startTime = sorted[0].time;
  const endTime = sorted[sorted.length - 1].time;
  const predictedStart = interceptMedian + median * startTime;
  const predictedEnd = interceptMedian + median * endTime;

  return {
    slope: parseFloat(median.toFixed(8)),
    intercept: parseFloat(interceptMedian.toFixed(6)),
    slope_per_year: parseFloat((median * 365.25 * 24 * 3600 * 1000).toFixed(6)),
    predicted_start: parseFloat(predictedStart.toFixed(4)),
    predicted_end: parseFloat(predictedEnd.toFixed(4)),
    n_slopes: slopes.length,
  };
}

async function computeStationTrend(stationId) {
  const result = await query(
    `SELECT observed_at, groundwater_level
     FROM telemetry_observations
     WHERE station_id = $1 AND groundwater_level IS NOT NULL
     ORDER BY observed_at ASC`,
    [stationId]
  );

  const rows = result.rows;
  if (rows.length < 4) {
    return {
      station_id: stationId,
      mann_kendall: null,
      sens_slope: null,
      observation_count: rows.length,
      message: "Insufficient data for trend analysis (need at least 4 observations)",
    };
  }

  const data = rows.map((r, i) => ({
    time: new Date(r.observed_at).getTime() / 1000,
    value: parseFloat(r.groundwater_level),
    timestamp: r.observed_at,
  }));

  const mk = mankendall(data);
  const ss = sensSlope(data);

  const firstObs = data[0].timestamp;
  const lastObs = data[data.length - 1].timestamp;

  return {
    station_id: stationId,
    mann_kendall: mk,
    sens_slope: ss,
    observation_count: data.length,
    period: { from: firstObs, to: lastObs },
  };
}

async function computeTrendSummary() {
  const stations = await query(
    `SELECT s.id, s.station_name, s.state, s.district, COUNT(t.id) AS obs_count
     FROM stations s
     JOIN telemetry_observations t ON t.station_id = s.id
     WHERE t.groundwater_level IS NOT NULL
     GROUP BY s.id, s.station_name, s.state, s.district
     HAVING COUNT(t.id) >= 4
     ORDER BY s.state, s.district`
  );

  const trends = [];
  for (const station of stations.rows) {
    const trend = await computeStationTrend(station.id);
    trends.push({
      station_id: station.id,
      station_name: station.station_name,
      state: station.state,
      district: station.district,
      ...trend,
    });
  }

  const increasing = trends.filter(t => t.mann_kendall && t.mann_kendall.direction === "increasing").length;
  const decreasing = trends.filter(t => t.mann_kendall && t.mann_kendall.direction === "decreasing").length;
  const noTrend = trends.filter(t => t.mann_kendall && t.mann_kendall.direction === "no trend").length;

  return {
    total_stations: trends.length,
    summary: { increasing, decreasing, no_trend: noTrend },
    trends,
  };
}

module.exports = { mankendall, sensSlope, computeStationTrend, computeTrendSummary };
