const assessmentService = require("../services/assessmentService");

async function getMany(req, res, next) {
  try {
    const { state, district, year, status, unitId, limit = 50, offset = 0 } = req.query;
    const result = await assessmentService.findRecords({
      state, district, year, status,
      unitId: unitId ? parseInt(unitId) : undefined,
      limit: parseInt(limit), offset: parseInt(offset),
    });
    res.json({ meta: { total: result.total, limit: parseInt(limit), offset: parseInt(offset) }, data: result.rows });
  } catch (err) { next(err); }
}

async function getById(req, res, next) {
  try {
    const record = await assessmentService.findRecordById(parseInt(req.params.id));
    if (!record) return res.status(404).json({ error: "Assessment record not found" });
    res.json(record);
  } catch (err) { next(err); }
}

async function getUnitHistory(req, res, next) {
  try {
    const { unitId } = req.params;
    const records = await assessmentService.getUnitHistory(parseInt(unitId));
    res.json({ data: records });
  } catch (err) { next(err); }
}

async function getSummary(req, res, next) {
  try {
    const { state, year } = req.query;
    const data = await assessmentService.getSummary({ state, year });
    res.json({ data });
  } catch (err) { next(err); }
}

async function getAvailableYears(req, res, next) {
  try {
    const data = await assessmentService.getAvailableYears();
    res.json({ data });
  } catch (err) { next(err); }
}

module.exports = { getMany, getById, getUnitHistory, getSummary, getAvailableYears };
