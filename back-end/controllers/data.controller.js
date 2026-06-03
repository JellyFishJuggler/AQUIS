const repo = require("../services/groundwaterRepository");

// ─────────────────────────────────────────────────────────
// GET /data
// Query params: state, district, status, limit, offset, sort_by, order
// ─────────────────────────────────────────────────────────
async function getAll(req, res, next) {
  try {
    const {
      state, district, status,
      limit  = 50,
      offset = 0,
      sort_by = "id",
      order   = "asc",
    } = req.query;

    const { rows, total } = await repo.findAll({
      state, district, status,
      limit:   Number(limit),
      offset:  Number(offset),
      sort_by, order,
    });

    res.json({
      meta: { total, limit: Number(limit), offset: Number(offset) },
      data: rows,
    });
  } catch (err) {
    next(err);
  }
}

// ─────────────────────────────────────────────────────────
// GET /data/:id
// ─────────────────────────────────────────────────────────
async function getOne(req, res, next) {
  try {
    const record = await repo.findById(req.params.id);
    if (!record) return res.status(404).json({ error: "Record not found" });
    res.json(record);
  } catch (err) {
    next(err);
  }
}

// ─────────────────────────────────────────────────────────
// POST /data
// ─────────────────────────────────────────────────────────
async function createOne(req, res, next) {
  try {
    const record = await repo.insertOne(req.body);
    res.status(201).json(record);
  } catch (err) {
    next(err);
  }
}

// ─────────────────────────────────────────────────────────
// PUT /data/:id  (partial update — send only fields to change)
// ─────────────────────────────────────────────────────────
async function updateOne(req, res, next) {
  try {
    const record = await repo.updateOne(req.params.id, req.body);
    if (!record) return res.status(404).json({ error: "Record not found" });
    res.json(record);
  } catch (err) {
    next(err);
  }
}

// ─────────────────────────────────────────────────────────
// DELETE /data/:id
// ─────────────────────────────────────────────────────────
async function deleteOne(req, res, next) {
  try {
    const deleted = await repo.deleteOne(req.params.id);
    if (!deleted) return res.status(404).json({ error: "Record not found" });
    res.json({ message: "Deleted successfully", id: deleted.id });
  } catch (err) {
    next(err);
  }
}

module.exports = { getAll, getOne, createOne, updateOne, deleteOne };
