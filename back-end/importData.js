const xlsx = require("xlsx");
const { Client } = require("pg");

const client = new Client({
  user: "postgres",
  host: "localhost",
  database: "aquis",
  password: "vanshika",
  port: 5432,
});

function getStatus(level) {
  if (level > 50000) return "SAFE";
  if (level > 20000) return "WARNING";
  return "CRITICAL";
}

async function importExcel() {
  await client.connect();

  const workbook = xlsx.readFile("src/dataset/ground_water_dataset.xlsx");
  const sheet = workbook.Sheets[workbook.SheetNames[2]]; // Public_View
  const data = xlsx.utils.sheet_to_json(sheet);

  // ================= STATE AVG =================
  const stateData = {};

  for (let row of data) {
    const state = row["STATE"];
    const rainfall = row["RAINFALL L_MM"];
    const recharge = row["ANNUAL_RECHARGE_HAM"]; // ✅ FIXED

    if (!state) continue;

    if (!stateData[state]) {
      stateData[state] = { rSum: 0, gSum: 0, count: 0 };
    }

    if (rainfall != null || recharge != null) {
      stateData[state].rSum += rainfall || 0;
      stateData[state].gSum += recharge || 0;
      stateData[state].count++;
    }
  }

  const stateAvg = {};

  for (let state in stateData) {
    const s = stateData[state];

    stateAvg[state] = {
      rainfall: s.count ? s.rSum / s.count : 0,
      recharge: s.count ? s.gSum / s.count : 0,
    };
  }

  // ================= FINAL INSERT =================
  for (let row of data) {
    const district = row["DISTRICT"];
    const state = row["STATE"];

    let rainfall = row["RAINFALL L_MM"];
    let recharge = row["ANNUAL_RECHARGE_HAM"]; // ✅ FIXED

    if (!district || !state) continue;

    // fallback
    if (rainfall == null) rainfall = stateAvg[state].rainfall;
    if (recharge == null) recharge = stateAvg[state].recharge;

    rainfall = rainfall || 0;
    recharge = recharge || 0;

    const area = 100000;
    const rainfall_HAM = (rainfall / 1000) * area;

    const water_level = (0.4 * rainfall_HAM) + (0.6 * recharge);

    const status = getStatus(water_level);

    await client.query(
      `INSERT INTO stations (district, state, water_level, status)
       VALUES ($1, $2, $3, $4)`,
      [district, state, water_level, status]
    );
  }

  console.log("✅ FINAL: Data imported correctly");
  await client.end();
}

importExcel();