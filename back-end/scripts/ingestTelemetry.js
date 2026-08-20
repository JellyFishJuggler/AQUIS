require("dotenv").config({ path: require("path").join(__dirname, "../.env") });
const { ingest } = require("../services/telemetryIngestion");

async function main() {
  const args = process.argv.slice(2);
  const limitIdx = args.indexOf("--limit");
  const limit = limitIdx >= 0 ? parseInt(args[limitIdx + 1]) : null;
  const batchIdx = args.indexOf("--batch");
  const batchSize = batchIdx >= 0 ? parseInt(args[batchIdx + 1]) : 100;

  const result = await ingest({ batchSize, limit });
  console.log("\nIngestion result:", JSON.stringify(result, null, 2));
}

main().catch(err => {
  console.error("Telemetry ingestion failed:", err.message);
  process.exit(1);
});
