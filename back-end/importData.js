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

    // ===== PASS 1: CALCULATE STATE AVERAGE =====
    for (let row of data) {
        const state = row["STATE"];
        const rainfall = row["RAINFALL L_MM"];
        const recharge = row["ANNUAL_RECHARGE_HAM"];

        if (!state) continue;

        if (!stateData[state]) {
            stateData[state] = { rSum: 0, gSum: 0, count: 0 };
        }

        if (rainfall != null && recharge != null) {
            stateData[state].rSum += rainfall;
            stateData[state].gSum += recharge;
            stateData[state].count++;
        }
    }

    // compute avg
    const stateAvg = {};

    for (let state in stateData) {
        const s = stateData[state];

        stateAvg[state] = {
            rainfall: s.count ? s.rSum / s.count : 0,
            recharge: s.count ? s.gSum / s.count : 0,
        };
    }

    // ================= FINAL INSERT =================
    let index = 1;

    for (let row of data) {
        const district = String(row["DISTRICT"] || "").trim();
        const state = String(row["STATE"] || "").trim();

        let rainfall = row["RAINFALL L_MM"];
        let recharge = row["ANNUAL_RECHARGE_HAM"];

        if (!district || !state) continue;

        if (rainfall == null && recharge == null) {
            rainfall = stateAvg[state].rainfall;
            recharge = stateAvg[state].recharge;
        }

        rainfall = rainfall ?? stateAvg[state].rainfall;
        recharge = recharge ?? stateAvg[state].recharge;

        const area = 100000;
        const rainfall_HAM = (rainfall / 1000) * area;

        const water_level = (0.4 * rainfall_HAM) + (0.6 * recharge);

        // 🔥 NEW: arbitrary station name
        const name = `Station-${index++}`;

        await client.query(
            `INSERT INTO stations (name, district, state, water_level, status)
     VALUES ($1, $2, $3, $4, 'UNKNOWN')`,
            [name, district, state, water_level]
        );
    }

    console.log("✅ FINAL: Data imported correctly");
    await client.end();
}

importExcel();