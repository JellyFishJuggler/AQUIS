const express = require("express");
const { Client } = require("pg");

const app = express();
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
// 📌 GET ALL + FILTER
// =======================
app.get("/stations", async (req, res) => {
  try {
    const { district, state } = req.query;

    let query = "SELECT * FROM stations";
    let values = [];

    if (district) {
      query += " WHERE district = $1";
      values.push(district);
    } else if (state) {
      query += " WHERE state = $1";
      values.push(state);
    }

    const result = await client.query(query, values);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// =======================
// 📌 GET SINGLE
// =======================
app.get("/stations/:id", async (req, res) => {
  try {
    const id = parseInt(req.params.id);

    const result = await client.query(
      "SELECT * FROM stations WHERE id = $1",
      [id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: "Not found" });
    }

    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// =======================
// 📌 DELETE
// =======================
app.delete("/stations/:id", async (req, res) => {
  try {
    const id = parseInt(req.params.id);

    await client.query("DELETE FROM stations WHERE id = $1", [id]);

    res.json({ message: "Station deleted" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// =======================
// 📌 UPDATE water_level
// =======================
app.put("/stations/:id", async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    const { water_level } = req.body;

    function getStatus(level) {
      if (level > 15) return "SAFE";
      if (level > 10) return "WARNING";
      return "CRITICAL";
    }

    const status = getStatus(water_level);

    const result = await client.query(
      `UPDATE stations 
       SET water_level = $1, status = $2
       WHERE id = $3 RETURNING *`,
      [water_level, status, id]
    );

    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// =======================
// 📊 ANALYTICS (BASIC)
// =======================
app.get("/analytics/avg", async (req, res) => {
  try {
    const result = await client.query(`
      SELECT district, AVG(water_level::float) as avg_water
      FROM stations
      WHERE water_level IS NOT NULL
      GROUP BY district
      ORDER BY avg_water DESC
    `);

    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});


app.get("/analytics/risky", async (req, res) => {
  try {
    const result = await client.query(`
      SELECT district, AVG(water_level) as avg_water
      FROM stations
      WHERE water_level IS NOT NULL
      GROUP BY district
      ORDER BY avg_water ASC
      LIMIT 5
    `);

    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


app.get("/alerts", async (req, res) => {
  try {
    const result = await client.query(`
      SELECT district, state, water_level, status
      FROM stations
      WHERE water_level < 20000
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


app.get("/map-data", async (req, res) => {
  try {
    const result = await client.query(`
      SELECT district, state, water_level, status
      FROM stations
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