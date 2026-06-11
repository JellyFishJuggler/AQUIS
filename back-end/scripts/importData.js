require("dotenv").config({ path: require("path").join(__dirname, "../.env") });
const XLSX = require("xlsx");
const { Pool } = require('pg');
const { classifyStatus } = require("../services/groundwaterClassification");

async function importData() {
  const pool = new Pool({ connectionString: process.env.DATABASE_URL });
  const client = await pool.connect();
  
  const SHEET_NAME = "Officer_view";
  
  try {
    console.log(`📂 Reading sheet: ${SHEET_NAME}...`);
    const workbook = XLSX.readFile('db/ground_water_dataset.xlsx');
    const worksheet = workbook.Sheets[SHEET_NAME];
    
    if (!worksheet) {
      console.error(`❌ Sheet "${SHEET_NAME}" not found!`);
      return;
    }
    
    const data = XLSX.utils.sheet_to_json(worksheet);
    console.log(`📊 Total rows found: ${data.length}`);

    await client.query('TRUNCATE TABLE groundwater_data;');
    console.log("🧹 Table truncated.");

    for (const row of data) {
      const extractionPct = parseFloat(row['Stage of Ground Water Extraction (%)_Total_Total']) || 0;
      const status = classifyStatus(extractionPct);

      await client.query(`
        INSERT INTO groundwater_data (
          state, district, assessment_unit, rainfall_mm, annual_recharge_ham, 
          total_extraction_ham, annual_extractable_gw_ham, extraction_rate_pct, status
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`, 
        [
          row.STATE || "N/A", 
          row.DISTRICT || "N/A", 
          row['ASSESSMENT UNIT'] || "N/A", 
          parseFloat(row.RAINFALL_MM) || 0, 
          parseFloat(row.ANNUAL_RECHARGE_HAM) || 0, 
          parseFloat(row['Ground Water Extraction for all uses (ha.m)_Total_Total']) || 0, 
          parseFloat(row['Annual Extractable Ground water Resource (ham)_Total_Total']) || 0, 
          extractionPct, 
          status
        ]
      );
    }
    
    console.log("🎉 Data successfully imported!");
  } catch (err) {
    console.error("❌ Error during import:", err.message);
  } finally {
    client.release();
    await pool.end();
  }
}

importData();