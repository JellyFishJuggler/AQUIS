const app      = require("./app");
const { pool } = require("./db/pool");

const PORT = process.env.PORT || 3000;

async function start() {
  // Verify DB connection before accepting traffic
  try {
    await pool.query("SELECT 1");
    console.log("✅ PostgreSQL connection verified");
  } catch (err) {
    console.error("❌ Cannot connect to PostgreSQL:", err.message);
    process.exit(1);
  }

  app.listen(PORT, () => {
    console.log(`🚀 AQUIS API running → http://localhost:${PORT}`);
  });
}

start();
