require("dotenv").config({ path: require("path").join(__dirname, "../.env") });
const { Pool } = require('pg');
const fs = require('fs');

async function importData() {
  const pool = new Pool({ connectionString: process.env.DATABASE_URL });
  
  try {
    // 1. Table saaf karo taaki purana data na ho
    await pool.query('TRUNCATE TABLE groundwater_data;');
    
    // 2. CSV file read karo (Excel ko pehle CSV mein save kar lena!)
    const csvData = fs.readFileSync('db/data.csv', 'utf8');
    
    // 3. PostgreSQL COPY command (Ye sabse fast hai)
    await pool.query(`
      COPY groundwater_data(state, district, assessment_unit, rainfall_mm, annual_recharge_ham, total_extraction_ham, annual_extractable_gw_ham, extraction_rate_pct, status) 
      FROM STDIN WITH (FORMAT csv, HEADER true)
    `);
    
    console.log("🎉 Data successfully imported!");
  } catch (err) {
    console.error("❌ Error:", err);
  } finally {
    await pool.end();
  }
}

importData();