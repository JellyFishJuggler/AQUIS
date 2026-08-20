require("dotenv").config({ path: require("path").join(__dirname, "../.env") });
const fs   = require("fs");
const path = require("path");
const { pool } = require("../db/pool");

async function migrate() {
  const sql = fs.readFileSync(path.join(__dirname, "../db/schema.sql"), "utf8");
  console.log("Running schema migration...");
  const statements = sql.split(";").filter(s => s.trim().length > 0);
  let applied = 0;
  for (const stmt of statements) {
    try {
      await pool.query(stmt);
      applied++;
    } catch (err) {
      if (err.code === "42710" || err.code === "42P07" || err.code === "42701") {
        continue;
      }
      console.error(`Warning: ${err.message} (${err.code})`);
    }
  }
  console.log(`Schema applied successfully (${applied} statements)`);
  await pool.end();
}

migrate().catch((err) => {
  console.error("Migration failed:", err.message);
  process.exit(1);
});
