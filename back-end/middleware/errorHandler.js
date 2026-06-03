/**
 * Express error-handling middleware (4-argument signature required).
 * All thrown errors bubble up here.
 */
// eslint-disable-next-line no-unused-vars
module.exports = function errorHandler(err, _req, res, _next) {
  const status = err.status || err.statusCode || 500;
  const message = err.message || "Internal Server Error";

  if (status >= 500) {
    console.error("[ERROR]", err.stack || err);
  }

  res.status(status).json({
    error: message,
    ...(process.env.NODE_ENV !== "production" && { stack: err.stack }),
  });
};
