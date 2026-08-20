const mlGateway = require("../services/mlGateway");
const modelService = require("../services/modelService");

async function getForecast(req, res, next) {
  try {
    const { stationId } = req.params;
    const selectedModel = await modelService.getSelected("forecast");

    if (!selectedModel) {
      return res.json({
        status: "unavailable",
        message: "No forecast model available yet",
        station_id: parseInt(stationId),
      });
    }

    const result = await mlGateway.getForecast(stationId, { model_id: selectedModel.id });
    if (!result.success) {
      return res.status(502).json({ error: "ML service error", detail: result.error });
    }
    res.json({
      station_id: parseInt(stationId),
      model: selectedModel.model_name,
      model_version: selectedModel.version,
      metrics: selectedModel.metrics,
      ...result.data,
    });
  } catch (err) { next(err); }
}

async function getAnomalies(req, res, next) {
  try {
    const { stationId } = req.params;
    if (stationId) {
      const result = await modelService.getStationAnomalies(parseInt(stationId), {
        limit: parseInt(req.query.limit) || 50,
        offset: parseInt(req.query.offset) || 0,
      });
      return res.json(result);
    }
    const { outputType, entityType, entityId, limit = 100, offset = 0 } = req.query;
    const result = await modelService.getOutputs({
      outputType: "anomaly",
      entityType, entityId: entityId ? parseInt(entityId) : undefined,
      limit: parseInt(limit), offset: parseInt(offset),
    });
    res.json(result);
  } catch (err) { next(err); }
}

async function getAnomalySummary(req, res, next) {
  try {
    const data = await modelService.getAnomalySummary();
    res.json({ data });
  } catch (err) { next(err); }
}

async function getRisk(req, res, next) {
  try {
    const { unitId } = req.params;
    const selectedModel = await modelService.getSelected("risk");

    if (!selectedModel) {
      return res.json({
        status: "unavailable",
        message: "No risk model available yet",
        unit_id: parseInt(unitId),
      });
    }

    const result = await mlGateway.getRisk(unitId, { model_id: selectedModel.id });
    if (!result.success) {
      return res.status(502).json({ error: "ML service error", detail: result.error });
    }
    res.json({
      unit_id: parseInt(unitId),
      model: selectedModel.model_name,
      model_version: selectedModel.version,
      ...result.data,
    });
  } catch (err) { next(err); }
}

async function getRiskSummary(req, res, next) {
  try {
    const outputs = await modelService.getOutputs({
      outputType: "risk", limit: 1000, offset: 0,
    });
    const riskCounts = {};
    outputs.rows.forEach(o => {
      const cat = o.risk_category || "unknown";
      riskCounts[cat] = (riskCounts[cat] || 0) + 1;
    });
    res.json({ data: { risk_distribution: riskCounts, total: outputs.total } });
  } catch (err) { next(err); }
}

async function getPriorityAreas(req, res, next) {
  try {
    const outputs = await modelService.getOutputs({
      outputType: "risk", limit: parseInt(req.query.limit) || 20, offset: 0,
    });
    const highRisk = outputs.rows.filter(o =>
      o.risk_category === "high" || o.risk_category === "critical"
    );
    res.json({ data: highRisk });
  } catch (err) { next(err); }
}

async function getModels(req, res, next) {
  try {
    const { task, limit = 50, offset = 0 } = req.query;
    const result = await modelService.findAll({ task, limit: parseInt(limit), offset: parseInt(offset) });
    res.json({ meta: { total: result.total }, data: result.rows });
  } catch (err) { next(err); }
}

async function getModelByName(req, res, next) {
  try {
    const models = await modelService.findAll({ limit: 1000 });
    const found = models.rows.find(m => m.model_name === req.params.modelName);
    if (!found) return res.status(404).json({ error: "Model not found" });
    res.json(found);
  } catch (err) { next(err); }
}

async function getModelMetrics(req, res, next) {
  try {
    const models = await modelService.findAll({ limit: 1000 });
    const found = models.rows.find(m => m.model_name === req.params.modelName);
    if (!found) return res.status(404).json({ error: "Model not found" });
    res.json({ model_name: found.model_name, version: found.version, metrics: found.metrics });
  } catch (err) { next(err); }
}

async function getModelComparison(req, res, next) {
  try {
    const { task } = req.query;
    if (!task) return res.status(400).json({ error: "task query parameter is required" });
    const comparison = await modelService.getComparison(task);
    res.json({ data: comparison });
  } catch (err) { next(err); }
}

async function getMLHealth(req, res, next) {
  try {
    const health = await mlGateway.healthCheck();
    res.json(health);
  } catch (err) { next(err); }
}

module.exports = {
  getForecast, getAnomalies, getAnomalySummary,
  getRisk, getRiskSummary, getPriorityAreas,
  getModels, getModelByName, getModelMetrics, getModelComparison,
  getMLHealth,
};
