/**
 * scripts/migrate.js
 * Runs schema.sql against the configured database.
 * Usage: node scripts/migrate.js
 */

require("dotenv").config({ path: require("path").join(__dirname, "../.env") });

const fs   = require("fs");
const path = require("path");
const { pool } = require("../db/pool");

async function migrate() {
  const sql = fs.readFileSync(path.join(__dirname, "../db/schema.sql"), "utf8");
  console.log("🔧 Running schema migration...");
  await pool.query(sql);
  console.log("✅ Schema applied successfully");
  await pool.end();
}

migrate().catch((err) => {
  console.error("❌ Migration failed:", err.message);
  process.exit(1);
});
