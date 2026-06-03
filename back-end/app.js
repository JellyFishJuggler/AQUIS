require("dotenv").config();
const express = require("express");
const cors    = require("cors");
const helmet  = require("helmet");
const morgan  = require("morgan");

const errorHandler   = require("./middleware/errorHandler");
const dataRoutes     = require("./routes/data.routes");
const analyticsRoutes = require("./routes/analytics.routes");
const alertRoutes    = require("./routes/alert.routes");

const app = express();

// ── Security & Parsing ──────────────────────────────────
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors());
app.use(express.json());
app.use(morgan(process.env.NODE_ENV === "production" ? "combined" : "dev"));

// ── Routes ───────────────────────────────────────────────
app.use("/data",      dataRoutes);
app.use("/analytics", analyticsRoutes);
app.use("/alerts",    alertRoutes);

// ── Health check ────────────────────────────────────────
app.get("/health", (_req, res) =>
  res.json({ status: "ok", timestamp: new Date().toISOString() })
);

// ── 404 ─────────────────────────────────────────────────
app.use((_req, res) => res.status(404).json({ error: "Route not found" }));

// ── Global error handler ─────────────────────────────────
app.use(errorHandler);

module.exports = app;
