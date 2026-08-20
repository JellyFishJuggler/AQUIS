require("dotenv").config();
const express = require("express");
const cors    = require("cors");
const helmet  = require("helmet");
const morgan  = require("morgan");

const errorHandler    = require("./middleware/errorHandler");
const dataRoutes      = require("./routes/data.routes");
const analyticsRoutes = require("./routes/analytics.routes");
const alertRoutes     = require("./routes/alert.routes");
const stationRoutes   = require("./routes/station.routes");
const telemetryRoutes = require("./routes/telemetry.routes");
const assessmentRoutes = require("./routes/assessment.routes");
const trendRoutes     = require("./routes/trend.routes");
const mlRoutes        = require("./routes/ml.routes");
const mlDataRoutes    = require("./routes/mlData.routes");
const dataQualityRoutes = require("./routes/dataQuality.routes");
const ingestionRoutes = require("./routes/ingestion.routes");

const app = express();

app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors());
app.use(express.json());
app.use(morgan(process.env.NODE_ENV === "production" ? "combined" : "dev"));

app.get("/health", (_req, res) =>
  res.json({ status: "ok", version: "2.0.0", timestamp: new Date().toISOString() })
);

app.use("/stations",      stationRoutes);
app.use("/telemetry",     telemetryRoutes);
app.use("/assessments",   assessmentRoutes);
app.use("/trends",        trendRoutes);
app.use("/ml",            mlRoutes);
app.use("/ml-data",       mlDataRoutes);
app.use("/data-quality",  dataQualityRoutes);
app.use("/ingestion",     ingestionRoutes);

app.use("/data",          dataRoutes);
app.use("/analytics",     analyticsRoutes);
app.use("/alerts",        alertRoutes);

app.use((_req, res) => res.status(404).json({ error: "Route not found" }));
app.use(errorHandler);

module.exports = app;
