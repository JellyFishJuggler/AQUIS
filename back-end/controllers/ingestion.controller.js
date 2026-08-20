const ingestionRunService = require("../services/ingestionRunService");

async function getRuns(req, res, next) {
  try {
    const { source, status, limit = 20, offset = 0 } = req.query;
    const result = await ingestionRunService.getRuns({
      source, status, limit: parseInt(limit), offset: parseInt(offset),
    });
    res.json({ meta: { total: result.total }, data: result.rows });
  } catch (err) { next(err); }
}

async function getRunById(req, res, next) {
  try {
    const run = await ingestionRunService.getRunById(parseInt(req.params.id));
    if (!run) return res.status(404).json({ error: "Ingestion run not found" });
    res.json(run);
  } catch (err) { next(err); }
}

async function triggerTelemetryIngestion(req, res, next) {
  try {
    const { batchSize, limit } = req.body || {};
    const telemetryIngestion = require("../services/telemetryIngestion");
    res.json({ message: "Telemetry ingestion started in background" });
    telemetryIngestion.ingest({ batchSize, limit }).catch(err => {
      console.error("Background telemetry ingestion failed:", err.message);
    });
  } catch (err) { next(err); }
}

async function triggerAssessmentIngestion(req, res, next) {
  try {
    const { sourceDir, assessmentYear } = req.body || {};
    const assessmentIngestion = require("../services/assessmentIngestion");
    res.json({ message: "Assessment ingestion started in background" });
    assessmentIngestion.ingestAll({ sourceDir, assessmentYear }).catch(err => {
      console.error("Background assessment ingestion failed:", err.message);
    });
  } catch (err) { next(err); }
}

module.exports = { getRuns, getRunById, triggerTelemetryIngestion, triggerAssessmentIngestion };
