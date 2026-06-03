const xlsx = require("xlsx");
const { Client } = require("pg");

const client = new Client({
  user: "postgres",
  host: "localhost",
  database: "aquis",
  password: "vanshika",
  port: 5432,
});

async function importExcel() {
  await client.connect();

  const workbook = xlsx.readFile("src/dataset/ground_water_dataset.xlsx");
  const sheet = workbook.Sheets[workbook.SheetNames[0]];

  const data = xlsx.utils.sheet_to_json(sheet);

  for (let row of data) {
    const state = row["STATE"];
    const district = row["DISTRICT"];
    const rainfall = row["RAINFALL"];
    const groundwater = row["GROUNDWATER"];

    if (!state || !district) continue;

    await client.query(
      "INSERT INTO stations (name, water_level, status) VALUES ($1, $2, $3)",
      [district, groundwater || 0, "UNKNOWN"]
    );
  }

  console.log("Data imported");
  await client.end();
}

importExcel();