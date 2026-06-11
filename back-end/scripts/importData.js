require("dotenv").config({ path: require("path").join(__dirname, "../.env") });
const XLSX = require("xlsx");
const path = require("path");
const { pool } = require("../db/pool");
const { classifyStatus } = require("../services/groundwaterClassification");

const EXCEL_PATH = process.argv[2] || path.join(__dirname, "../db/ground_water_dataset.xlsx");
const SHEET_NAME = "Officer_view";

async function main() {
  console.log(`📂 Reading: ${EXCEL_PATH}`);
  let workbook;
  try {
    workbook = XLSX.readFile(EXCEL_PATH);
  } catch (err) {
    console.error("❌ Cannot read file:", err.message);
    return;
  }

  const sheet = workbook.Sheets[SHEET_NAME];
  const rawRows = XLSX.utils.sheet_to_json(sheet, { defval: null });
  
  // Clean and prepare data
  const records = rawRows.map(row => ({
    state: String(row["STATE"] || "").trim(),
    district: String(row["DISTRICT"] || "").trim(),
    assessment_unit: String(row["ASSESSMENT UNIT"] || ""),
    rainfall_mm: parseFloat(row["RAINFALL_MM"]) || 0,
    annual_recharge_ham: parseFloat(row["ANNUAL_RECHARGE_HAM"]) || 0,
    total_extraction_ham: parseFloat(row["Ground Water Extraction for all uses (ha.m)_Total_Total"]) || 0,
    annual_extractable_gw_ham: parseFloat(row["Annual Extractable Ground water Resource (ham)_Total_Total"]) || 0,
    extraction_rate_pct: parseFloat(row["Stage of Ground Water Extraction (%)_Total_Total"]) || 0,
    status: classifyStatus(parseFloat(row["Stage of Ground Water Extraction (%)_Total_Total"]) || 0)
  })).filter(r => r.state && r.district);

  console.log(`✅ Records ready: ${records.length}. Inserting in batches...`);
  
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const BATCH_SIZE = 50;
    for (let i = 0; i < records.length; i += BATCH_SIZE) {
      const batch = records.slice(i, i + BATCH_SIZE);
      console.log(`Processing batch ${Math.floor(i/BATCH_SIZE) + 1}...`);
      for (const rec of batch) {
        await client.query(
          `INSERT INTO groundwater_data (state, district, assessment_unit, rainfall_mm, annual_recharge_ham, total_extraction_ham, annual_extractable_gw_ham, extraction_rate_pct, status) 
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
          [rec.state, rec.district, rec.assessment_unit, rec.rainfall_mm, rec.annual_recharge_ham, rec.total_extraction_ham, rec.annual_extractable_gw_ham, rec.extraction_rate_pct, rec.status]
        );
      }
    }
    await client.query("COMMIT");
    console.log("🎉 Data successfully imported!");
  } catch (err) {
    await client.query("ROLLBACK");
    console.error("❌ Error during insertion:", err.message);
  } finally {
    client.release();
    await pool.end();
  }
}

main().catch((err) => console.error("Fatal error:", err));