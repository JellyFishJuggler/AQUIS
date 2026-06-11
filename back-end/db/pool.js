const { Pool } = require("pg");
require("dotenv").config();

// Neon aur Render ke liye optimal connection config
const pool = new Pool({
  // Agar Render par DATABASE_URL set hai, toh use priority milegi
  connectionString: process.env.DATABASE_URL || undefined,
  
  // Local development ke liye fallback options
  host: process.env.DB_HOST || "localhost",
  port: parseInt(process.env.DB_PORT) || 5432,
  user: process.env.DB_USER || "postgres",
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME || "aquis",
  
  // Connection pooling settings
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
  
  // SSL Configuration: Neon.tech ke liye zaroori
  ssl: {
    rejectUnauthorized: false
  }
});

pool.on("error", (err) => {
  console.error("Unexpected error on idle DB client:", err.message);
});

// Lightweight health-check wrapper
const query = async (text, params) => {
  const start = Date.now();
  try {
    const res = await pool.query(text, params);
    const duration = Date.now() - start;
    if (process.env.NODE_ENV === "development") {
      console.log(`[DB] ${text.slice(0, 80)}… | rows: ${res.rowCount} | ${duration}ms`);
    }
    return res;
  } catch (err) {
    console.error("[DB ERROR]", err.message, "\nQuery:", text);
    throw err;
  }
};

module.exports = { pool, query };