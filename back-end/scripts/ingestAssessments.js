require("dotenv").config({ path: require("path").join(__dirname, "../.env") });
const { ingestAll } = require("../services/assessmentIngestion");

async function main() {
  const args = process.argv.slice(2);
  const yearIdx = args.indexOf("--year");
  const assessmentYear = yearIdx >= 0 ? args[yearIdx + 1] : null;
  const dirIdx = args.indexOf("--dir");
  const sourceDir = dirIdx >= 0 ? args[dirIdx + 1] : undefined;

  const result = await ingestAll({ sourceDir, assessmentYear });
  console.log("\nIngestion result:", JSON.stringify(result, null, 2));
}

main().catch(err => {
  console.error("Assessment ingestion failed:", err.message);
  process.exit(1);
});
