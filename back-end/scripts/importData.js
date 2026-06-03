/**
 * scripts/importData.js
 *
 * Reads Officer_view sheet from the CGWB DWLR Excel file,
 * cleans the data, derives extraction rate + status per the research paper,
 * and bulk-inserts into PostgreSQL.
 *
 * Usage:
 *   node scripts/importData.js [path/to/ground_water_dataset.xlsx]
 */

require("dotenv").config({ path: require("path").join(__dirname, "../.env") });

const XLSX   = require("xlsx");
const path   = require("path");
const { pool } = require("../db/pool");
const { deriveClassification } = require("../services/groundwaterClassification");

// ── Config ───────────────────────────────────────────────
const EXCEL_PATH = process.argv[2] ||
  path.join(__dirname, "../dataset/ground_water_dataset.xlsx");

const SHEET_NAME = "Officer_view";  // has extraction data needed for classification

// ── Column map: Excel header → our DB field ───────────────
// Handles the verbose CGWB column names
const COL = {
  STATE:                      "STATE",
  DISTRICT:                   "DISTRICT",
  ASSESSMENT_UNIT:            "ASSESSMENT UNIT",
  RAINFALL_MM:                "RAINFALL_MM",
  RECHARGE_WORTHY_AREA_HA:    "RECHARGE_WORTHY_AREA_HA",
  RECHARGE_RAINFALL_HAM:      "RECHARGE_RAINFALL_HAM",
  RECHARGE_CANALS_HAM:        "RECHARGE_CANALS_HAM",
  RECHARGE_SURFACE_IRR_HAM:   "RECHARGE_SURFACE_IRR_HAM",
  RECHARGE_GW_IRR_HAM:        "RECHARGE_GW_IRR_HAM",
  RECHARGE_TANKS_PONDS_HAM:   "RECHARGE_TANKS_PONDS_HAM",
  RECHARGE_WCS_HAM:           "RECHARGE_WCS_HAM",
  RECHARGE_PIPELINES_HAM:     "RECHARGE_PIPELINES_HAM",
  RECHARGE_SEWAGE_FF_HAM:     "RECHARGE_SEWAGE_FF_HAM",
  TOTAL_RECHARGE_HAM:         "TOTAL_RECHARGE_HAM",
  ANNUAL_RECHARGE_HAM:        "ANNUAL_RECHARGE_HAM",
  ENV_FLOWS_HAM:              "Environmental Flows (ham)_Total_Total",
  ANNUAL_EXTRACTABLE_GW_HAM:  "Annual Extractable Ground water Resource (ham)_Total_Total",
  EXTRACTION_DOMESTIC:        "Ground Water Extraction for all uses (ha.m)_Domestic_Total",
  EXTRACTION_INDUSTRIAL:      "Ground Water Extraction for all uses (ha.m)_Industrial_Total",
  EXTRACTION_IRRIGATION:      "Ground Water Extraction for all uses (ha.m)_Irrigation_Total",
  TOTAL_EXTRACTION:           "Ground Water Extraction for all uses (ha.m)_Total_Total",
  EXTRACTION_STAGE_PCT:       "Stage of Ground Water Extraction (%)_Total_Total",
  NET_AVAILABILITY_FUTURE:    "Net Annual Ground Water Availability for Future Use (ham)_Total_Total",
  STORAGE_UNCONFINED_FRESH:   "In-Storage Unconfined Ground Water Resources(ham)_Other Parameters Present_Fresh",
  AVAIL_UNCONFINED_FRESH:     "Total Ground Water Availability in Unconfined Aquifier (ham)_Other Parameters Present_Fresh",
  AVAIL_CONFINED_FRESH:       "Total Confined Ground Water Resources (ham)_Other Parameters Present_Fresh",
  AVAIL_SEMICONFINED_FRESH:   "Total Semi-Confined Ground Water Resources (ham)_Other Parameters Present_Fresh",
  TOTAL_GW_AVAIL:             "Total Ground Water Availability in the area (ham)_Other Parameters Present_Fresh",
};

// ── Helpers ───────────────────────────────────────────────

function safeFloat(v) {
  const n = parseFloat(v);
  if (isNaN(n) || !isFinite(n)) return null;
  return n;
}

function safeStr(v) {
  if (v == null) return null;
  const s = String(v).trim();
  return s === "" || s === "0" ? null : s;
}

/**
 * Compute state-level averages for imputation of missing values.
 * (Your importData.js did this — kept and improved here.)
 */
function buildStateAverages(rows) {
  const acc = {};

  for (const row of rows) {
    const state = String(row[COL.STATE] || "").trim();
    if (!state || state === "0") continue;

    const rainfall  = safeFloat(row[COL.RAINFALL_MM]);
    const recharge  = safeFloat(row[COL.ANNUAL_RECHARGE_HAM]);
    const extr      = safeFloat(row[COL.TOTAL_EXTRACTION]);
    const extractable = safeFloat(row[COL.ANNUAL_EXTRACTABLE_GW_HAM]);

    if (!acc[state]) {
      acc[state] = { rSum: 0, rechargeSum: 0, extrSum: 0, extractableSum: 0, n: 0 };
    }

    const s = acc[state];
    if (rainfall    != null) { s.rSum         += rainfall;    s.n++; }
    if (recharge    != null)   s.rechargeSum  += recharge;
    if (extr        != null)   s.extrSum      += extr;
    if (extractable != null)   s.extractableSum += extractable;
  }

  const avgs = {};
  for (const [state, s] of Object.entries(acc)) {
    avgs[state] = {
      rainfall:    s.n ? s.rSum / s.n : 0,
      recharge:    s.n ? s.rechargeSum / s.n : 0,
      extraction:  s.n ? s.extrSum / s.n : 0,
      extractable: s.n ? s.extractableSum / s.n : 0,
    };
  }
  return avgs;
}

// ── Main ─────────────────────────────────────────────────

async function main() {
  console.log(`📂 Reading: ${EXCEL_PATH}`);

  let workbook;
  try {
    workbook = XLSX.readFile(EXCEL_PATH);
  } catch (err) {
    console.error("❌ Cannot open Excel file:", err.message);
    process.exit(1);
  }

  if (!workbook.SheetNames.includes(SHEET_NAME)) {
    console.error(`❌ Sheet "${SHEET_NAME}" not found. Available:`, workbook.SheetNames);
    process.exit(1);
  }

  const sheet = workbook.Sheets[SHEET_NAME];
  const rawRows = XLSX.utils.sheet_to_json(sheet, { defval: null });
  console.log(`📊 Raw rows loaded: ${rawRows.length}`);

  // Build state-level averages for imputation
  const stateAvg = buildStateAverages(rawRows);

  // ── Transform + clean ─────────────────────────────────
  const records = [];
  let skippedNoLocation = 0;
  let skippedZeroRow = 0;

  for (const row of rawRows) {
    const state    = String(row[COL.STATE]    || "").trim();
    const district = String(row[COL.DISTRICT] || "").trim();

    // Skip header-continuation rows (all zeros) and missing location
    if (!state || !district || state === "0" || district === "0") {
      skippedNoLocation++;
      continue;
    }

    // Detect and skip the all-zero junk row at top of sheet
    const totalExtRaw = safeFloat(row[COL.TOTAL_EXTRACTION]);
    const annualRechargeRaw = safeFloat(row[COL.ANNUAL_RECHARGE_HAM]);
    if (totalExtRaw === 0 && annualRechargeRaw === 0 &&
        safeFloat(row[COL.RAINFALL_MM]) === 0) {
      skippedZeroRow++;
      continue;
    }

    const avg = stateAvg[state] || {};

    // Impute with state average where null
    const rainfall  = safeFloat(row[COL.RAINFALL_MM])          ?? avg.rainfall    ?? null;
    const recharge  = safeFloat(row[COL.ANNUAL_RECHARGE_HAM])  ?? avg.recharge    ?? null;
    const extr      = safeFloat(row[COL.TOTAL_EXTRACTION])      ?? avg.extraction  ?? null;
    const extractable = safeFloat(row[COL.ANNUAL_EXTRACTABLE_GW_HAM]) ?? avg.extractable ?? null;

    // Prefer dataset's pre-computed extraction % if present, else derive
    let extractionRatePct = safeFloat(row[COL.EXTRACTION_STAGE_PCT]);
    let status;

    if (extractionRatePct != null && extractionRatePct > 0) {
      // Cap outliers per paper §6.1
      extractionRatePct = Math.min(extractionRatePct, 300);
      const { classifyStatus } = require("../services/groundwaterClassification");
      status = classifyStatus(extractionRatePct);
    } else {
      // Derive from raw extraction / extractable
      const derived = deriveClassification(extr, extractable);
      extractionRatePct = derived.extractionRatePct;
      status = derived.status;
    }

    records.push({
      state,
      district,
      assessment_unit:              safeStr(row[COL.ASSESSMENT_UNIT]),
      rainfall_mm:                  rainfall,
      recharge_worthy_area_ha:      safeFloat(row[COL.RECHARGE_WORTHY_AREA_HA]),
      recharge_rainfall_ham:        safeFloat(row[COL.RECHARGE_RAINFALL_HAM]),
      recharge_canals_ham:          safeFloat(row[COL.RECHARGE_CANALS_HAM]),
      recharge_surface_irr_ham:     safeFloat(row[COL.RECHARGE_SURFACE_IRR_HAM]),
      recharge_gw_irr_ham:          safeFloat(row[COL.RECHARGE_GW_IRR_HAM]),
      recharge_tanks_ponds_ham:     safeFloat(row[COL.RECHARGE_TANKS_PONDS_HAM]),
      recharge_wcs_ham:             safeFloat(row[COL.RECHARGE_WCS_HAM]),
      recharge_pipelines_ham:       safeFloat(row[COL.RECHARGE_PIPELINES_HAM]),
      recharge_sewage_ff_ham:       safeFloat(row[COL.RECHARGE_SEWAGE_FF_HAM]),
      total_recharge_ham:           safeFloat(row[COL.TOTAL_RECHARGE_HAM]),
      stream_recharge_ham:          null,  // not in Officer_view
      annual_recharge_ham:          recharge,
      environmental_flows_ham:      safeFloat(row[COL.ENV_FLOWS_HAM]),
      annual_extractable_gw_ham:    extractable,
      extraction_domestic_ham:      safeFloat(row[COL.EXTRACTION_DOMESTIC]),
      extraction_industrial_ham:    safeFloat(row[COL.EXTRACTION_INDUSTRIAL]),
      extraction_irrigation_ham:    safeFloat(row[COL.EXTRACTION_IRRIGATION]),
      total_extraction_ham:         extr,
      extraction_rate_pct:          extractionRatePct,
      net_availability_future_ham:  safeFloat(row[COL.NET_AVAILABILITY_FUTURE]),
      storage_unconfined_fresh_ham: safeFloat(row[COL.STORAGE_UNCONFINED_FRESH]),
      availability_unconfined_fresh: safeFloat(row[COL.AVAIL_UNCONFINED_FRESH]),
      availability_confined_fresh:  safeFloat(row[COL.AVAIL_CONFINED_FRESH]),
      availability_semiconfined_fresh: safeFloat(row[COL.AVAIL_SEMICONFINED_FRESH]),
      total_gw_availability_ham:    safeFloat(row[COL.TOTAL_GW_AVAIL]),
      status,
    });
  }

  console.log(`✅ Clean records ready: ${records.length}`);
  console.log(`   Skipped (no location): ${skippedNoLocation}`);
  console.log(`   Skipped (zero rows):   ${skippedZeroRow}`);

  if (records.length === 0) {
    console.error("❌ No records to insert.");
    process.exit(1);
  }

  // ── Show classification preview ────────────────────────
  const preview = { SAFE: 0, SEMI_CRITICAL: 0, CRITICAL: 0, OVER_EXPLOITED: 0, null: 0 };
  for (const r of records) preview[r.status ?? "null"]++;
  console.log("📊 Status distribution preview:", preview);

  // ── Bulk insert ───────────────────────────────────────
  console.log("💾 Inserting into PostgreSQL...");
  const client = await pool.connect();
  let inserted = 0;
  let failed   = 0;

  try {
    await client.query("BEGIN");

    for (const rec of records) {
      try {
        await client.query(
          `INSERT INTO groundwater_data (
            state, district, assessment_unit,
            rainfall_mm,
            recharge_worthy_area_ha, recharge_rainfall_ham, recharge_canals_ham,
            recharge_surface_irr_ham, recharge_gw_irr_ham, recharge_tanks_ponds_ham,
            recharge_wcs_ham, recharge_pipelines_ham, recharge_sewage_ff_ham,
            total_recharge_ham, stream_recharge_ham, annual_recharge_ham,
            environmental_flows_ham, annual_extractable_gw_ham,
            extraction_domestic_ham, extraction_industrial_ham, extraction_irrigation_ham,
            total_extraction_ham, extraction_rate_pct, net_availability_future_ham,
            storage_unconfined_fresh_ham, availability_unconfined_fresh,
            availability_confined_fresh, availability_semiconfined_fresh,
            total_gw_availability_ham, status
          ) VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,
            $17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30
          )`,
          [
            rec.state, rec.district, rec.assessment_unit,
            rec.rainfall_mm,
            rec.recharge_worthy_area_ha, rec.recharge_rainfall_ham, rec.recharge_canals_ham,
            rec.recharge_surface_irr_ham, rec.recharge_gw_irr_ham, rec.recharge_tanks_ponds_ham,
            rec.recharge_wcs_ham, rec.recharge_pipelines_ham, rec.recharge_sewage_ff_ham,
            rec.total_recharge_ham, rec.stream_recharge_ham, rec.annual_recharge_ham,
            rec.environmental_flows_ham, rec.annual_extractable_gw_ham,
            rec.extraction_domestic_ham, rec.extraction_industrial_ham, rec.extraction_irrigation_ham,
            rec.total_extraction_ham, rec.extraction_rate_pct, rec.net_availability_future_ham,
            rec.storage_unconfined_fresh_ham, rec.availability_unconfined_fresh,
            rec.availability_confined_fresh, rec.availability_semiconfined_fresh,
            rec.total_gw_availability_ham, rec.status,
          ]
        );
        inserted++;
      } catch (rowErr) {
        console.warn(`  ⚠️  Row skipped (${rec.district}, ${rec.state}):`, rowErr.message);
        failed++;
      }
    }

    await client.query("COMMIT");
  } catch (err) {
    await client.query("ROLLBACK");
    console.error("❌ Transaction rolled back:", err.message);
    process.exit(1);
  } finally {
    client.release();
    await pool.end();
  }

  console.log(`\n✅ Import complete`);
  console.log(`   Inserted: ${inserted}`);
  console.log(`   Failed:   ${failed}`);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
