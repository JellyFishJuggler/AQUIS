const https = require("https");
const http = require("http");
const { pool } = require("../db/pool");
const ingestionRunService = require("./ingestionRunService");
const stationService = require("./stationService");

const NWDP_API_BASE = "https://nwdp.nwic.gov.in/api/3/action/datastore_search";
const RESOURCE_ID = "84bfda45-8ead-436d-9c8e-f7a93ee57522";
const DEFAULT_BATCH_SIZE = 100;
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;
const REQUEST_TIMEOUT_MS = 30000;

function fetchPage(url, timeout = REQUEST_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https") ? https : http;
    const req = client.get(url, { timeout }, (res) => {
      let data = "";
      res.on("data", (chunk) => { data += chunk; });
      res.on("end", () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(new Error(`JSON parse error: ${e.message}`));
        }
      });
    });
    req.on("timeout", () => { req.destroy(); reject(new Error("Request timeout")); });
    req.on("error", reject);
  });
}

function parseTimestamp(ts) {
  if (!ts || ts === "-") return null;
  const cleaned = String(ts).trim();
  const formats = [
    /^(\d{2})-(\d{2})-(\d{4})\s+(\d{2}):(\d{2})$/,
    /^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})$/,
  ];
  const m1 = cleaned.match(formats[0]);
  if (m1) {
    const d = new Date(`${m1[3]}-${m1[2]}-${m1[1]}T${m1[4]}:${m1[5]}:00Z`);
    return isNaN(d.getTime()) ? null : d.toISOString();
  }
  const m2 = cleaned.match(formats[1]);
  if (m2) {
    const d = new Date(`${m2[1]}-${m2[2]}-${m2[3]}T${m2[4]}:${m2[5]}:00Z`);
    return isNaN(d.getTime()) ? null : d.toISOString();
  }
  const d = new Date(cleaned);
  return isNaN(d.getTime()) ? null : d.toISOString();
}

function parseNumeric(val) {
  if (val == null || val === "-" || val === "" || val === "null") return null;
  const cleaned = String(val).trim();
  const num = parseFloat(cleaned);
  return isNaN(num) ? null : num;
}

function cleanText(val) {
  if (val == null) return null;
  const cleaned = String(val).trim();
  if (cleaned === "-" || cleaned === "" || cleaned.toLowerCase() === "na" || cleaned.toLowerCase() === "n/a") return null;
  return cleaned;
}

function extractStationKey(record) {
  return cleanText(record.Station) || `unknown_${record._id || Date.now()}`;
}

async function withRetry(fn, retries = MAX_RETRIES, delay = RETRY_DELAY_MS) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt === retries) throw err;
      console.log(`  Retry ${attempt}/${retries} after ${delay}ms: ${err.message}`);
      await new Promise((r) => setTimeout(r, delay * attempt));
    }
  }
}

async function ingest({ batchSize = DEFAULT_BATCH_SIZE, limit = null, incremental = true } = {}) {
  const runId = await ingestionRunService.createRun("telemetry_api", RESOURCE_ID, {
    batchSize, limit, incremental,
  });
  console.log(`\n🛰️  Telemetry ingestion started (run ${runId})`);

  const stats = {
    recordsSeen: 0,
    recordsInserted: 0,
    recordsDuplicates: 0,
    recordsRejected: 0,
    errorCount: 0,
    errors: [],
  };

  let offset = 0;
  let totalAvailable = null;
  let done = false;

  try {
    while (!done) {
      const url = `${NWDP_API_BASE}?resource_id=${RESOURCE_ID}&limit=${batchSize}&offset=${offset}`;
      console.log(`  Fetching offset=${offset}...`);

      const response = await withRetry(() => fetchPage(url));
      if (!response.success || !response.result) {
        throw new Error("API returned unsuccessful response");
      }

      const records = response.result.records || [];
      if (totalAvailable === null) {
        totalAvailable = response.result.total || 0;
        console.log(`  Total available records: ${totalAvailable}`);
      }

      if (records.length === 0) break;

      const client = await pool.connect();
      try {
        await client.query("BEGIN");

        for (const record of records) {
          stats.recordsSeen++;

          try {
            const stationKey = extractStationKey(record);
            const stationId = await stationService.upsertStation({
              external_station_id: stationKey,
              station_name: cleanText(record.Station),
              agency: cleanText(record.Agency),
              state: cleanText(record.State),
              state_lgd_code: cleanText(record["State LGD Code"]),
              district: cleanText(record.District),
              district_lgd_code: cleanText(record["District LGD Code"]),
              tehsil: cleanText(record.Tehsil),
              block: cleanText(record.Block),
              village: cleanText(record.Village),
              river: cleanText(record.River),
              basin: cleanText(record.Basin),
              tributary: cleanText(record.Tributary),
              subtributary: cleanText(record.Subtributary),
              sub_subtributary: cleanText(record["SubSubtributary"]),
              local_river: cleanText(record["Local River"]),
              latitude: parseNumeric(record.Latitude),
              longitude: parseNumeric(record.Longitude),
              rl_msl: parseNumeric(record.RL_MSL),
            });

            const observedAt = parseTimestamp(record["Data Acquisition Time"]);
            if (!observedAt) {
              stats.recordsRejected++;
              continue;
            }

            const gwLevel = parseNumeric(record["Groundwater Level Telemetry 6 Hourly (meter)"]);
            const sourceRecordId = String(record._id || "");

            const insertResult = await client.query(
              `INSERT INTO telemetry_observations
                 (station_id, observed_at, groundwater_level, source, source_record_id, ingestion_run_id)
               VALUES ($1, $2, $3, 'nwdp_api', $4, $5)
               ON CONFLICT (station_id, observed_at) DO NOTHING
               RETURNING id`,
              [stationId, observedAt, gwLevel, sourceRecordId, runId]
            );

            if (insertResult.rows.length > 0) {
              stats.recordsInserted++;
              await client.query(
                `UPDATE stations SET
                   first_observed_at = LEAST(COALESCE(first_observed_at, $2), $2),
                   last_observed_at = GREATEST(COALESCE(last_observed_at, $2), $2),
                   observation_count = observation_count + 1
                 WHERE id = $1`,
                [stationId, observedAt]
              );
            } else {
              stats.recordsDuplicates++;
            }
          } catch (rowErr) {
            stats.errorCount++;
            stats.errors.push({ offset, error: rowErr.message });
            if (stats.errors.length > 100) stats.errors = stats.errors.slice(-50);
          }
        }

        await client.query("COMMIT");
      } catch (batchErr) {
        await client.query("ROLLBACK");
        throw batchErr;
      } finally {
        client.release();
      }

      offset += batchSize;
      if (limit && stats.recordsSeen >= limit) break;
      if (offset >= totalAvailable) break;

      await new Promise((r) => setTimeout(r, 500));
    }

    await ingestionRunService.finishRun(runId, {
      status: "completed",
      ...stats,
      errorSummary: stats.errors.length > 0 ? { samples: stats.errors.slice(0, 20) } : null,
    });

    console.log(`✅ Telemetry ingestion complete: ${stats.recordsInserted} inserted, ${stats.recordsDuplicates} duplicates, ${stats.recordsRejected} rejected, ${stats.errorCount} errors`);
    return { runId, stats };
  } catch (err) {
    await ingestionRunService.finishRun(runId, {
      status: "failed",
      ...stats,
      errorSummary: { fatal: err.message, samples: stats.errors.slice(0, 20) },
    });
    console.error(`❌ Telemetry ingestion failed: ${err.message}`);
    throw err;
  }
}

module.exports = { ingest, fetchPage, parseTimestamp, parseNumeric, cleanText };
