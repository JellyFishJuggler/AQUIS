const express = require("express");
const cors = require("cors");
const { Client } = require("pg");

const app = express();

app.use(cors());
app.use(express.json());

// DB connection
const client = new Client({
  user: "postgres",
  host: "localhost",
  database: "aquis",
  password: "vanshika",
  port: 5432,
});

client
  .connect()
  .then(() => console.log("✅ Connected to PostgreSQL"))
  .catch((err) => console.error("❌ DB Connection Error", err));


// =======================
// 🔥 DYNAMIC STATUS LOGIC
// =======================
async function updateDynamicStatus() {
  const result = await client.query(`
    SELECT id, water_level FROM stations
    WHERE water_level IS NOT NULL
    ORDER BY water_level ASC
  `);

  const values = result.rows.map(r => r.water_level);

  if (values.length === 0) return;

  const p30 = values[Math.floor(values.length * 0.3)];
  const p70 = values[Math.floor(values.length * 0.7)];

  console.log("📊 Thresholds:", { p30, p70 });

  for (let row of result.rows) {
    let status = "CRITICAL";

    if (row.water_level >= p70) status = "SAFE";
    else if (row.water_level >= p30) status = "WARNING";

    await client.query(
      `UPDATE stations SET status = $1 WHERE id = $2`,
      [status, row.id]
    );
  }

  console.log("✅ Dynamic status updated");
}


// =======================
// 📌 TRIGGER STATUS UPDATE
// =======================
app.get("/update-status", async (req, res) => {
  try {
    await updateDynamicStatus();
    res.json({ message: "Dynamic status updated" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// =======================
// 📌 GET ALL + FILTER
// =======================
app.get("/stations", async (req, res) => {
  try {
    const { district, state } = req.query;

    let query = "SELECT * FROM stations";
    let values = [];

    if (district) {
      query += " WHERE district ILIKE $1";
      values.push(`%${district}%`);
    } else if (state) {
      query += " WHERE state ILIKE $1";
      values.push(`%${state}%`);
    }

    const result = await client.query(query, values);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// =======================
// 📊 ANALYTICS
// =======================
app.get("/analytics/avg", async (req, res) => {
  try {
    const result = await client.query(`
      SELECT district, AVG(water_level) as avg_water
      FROM stations
      WHERE district IS NOT NULL
      GROUP BY district
      ORDER BY avg_water DESC
    `);

    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


app.get("/analytics/risky", async (req, res) => {
  try {
    const result = await client.query(`
      SELECT district, state, AVG(water_level) as avg_water
      FROM stations
      GROUP BY district, state
      ORDER BY avg_water ASC
      LIMIT 5
    `);

    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// =======================
// 🚨 ALERTS
// =======================
app.get("/alerts", async (req, res) => {
  try {
    const result = await client.query(`
      SELECT district, state, water_level, status
      FROM stations
      ORDER BY water_level ASC
      LIMIT 10
    `);

    res.json({
      alert: "Critical groundwater levels",
      districts: result.rows
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// =======================
// 📊 STATE SUMMARY
// =======================
app.get("/analytics/state-summary", async (req, res) => {
  try {
    const result = await client.query(`
      SELECT state, 
             AVG(water_level) as avg_water,
             MIN(water_level) as min_water,
             MAX(water_level) as max_water
      FROM stations
      GROUP BY state
      ORDER BY avg_water DESC
    `);

    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// =======================
app.listen(3000, () => {
  console.log("🚀 Server running on port 3000");
});