

const { Client } = require("pg")

const client = new Client({
  user: "postgres",
  host: "localhost",
  database: "aquis",
  password: "vanshika",
  port: 5432,
})

client.connect().then(() =>  console.log("Connected to postgres")).catch(err => console.error("Connection Error", err))


const express = require("express");
const app = express();

app.use(express.json());

function getStatus(level) {
  if (level > 15) return "SAFE";
  if (level > 10) return "WARNING";
  return "CRITICAL";
}


let stations = [
  {
    id: 1,
    name: "Station-001",
    water_level: 15.2,
    status: "SAFE"
  }
];

app.get("/stations", async (req, res) => {
  //res.json(stations);
  
  try {
    const result = await client.query("SELECT * FROM stations")
    res.json(result.rows)
  } catch (err) {
    console.error(err);
    res.status(500).json({error: "DB error"})
  }

});

app.get("/stations/:id", (req, res) => {
  const id = parseInt(req.params.id);

  const station = stations.find(s => s.id === id);

  if (!station) {
    return res.status(404).json({
      error: "Station not found"
    });
  }

  res.json(station);
});


app.post("/stations", async (req, res) => {

  const { name, water_level } = req.body;
  const status = getStatus(water_level);

  try {
    const result = await client.query(
      "INSERT INTO stations (name, water_level, status) VALUES ($1, $2, $3) RETURNING *",
      [name,water_level,status]);

    res.json({
      message: "Station added successfully",
      data: result.rows[0],
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({error: "DB error"})
  }
});

app.delete("/stations/:id", (req, res) => {
  const id = parseInt(req.params.id);

  const index = stations.findIndex(s => s.id === id);

  if (index === -1) {
    return res.status(404).json({
      error: "Station not found"
    });
  }

  stations.splice(index, 1);

  res.json({
    message: "Station deleted successfully"
  });
});


app.put("/stations/:id", (req, res) => {
  const id = parseInt(req.params.id)
  const {water_level} = req.body;

  const station = stations.find(s => s.id === id)

  if(!station){
    return res.status(404).json({error: "Not Found"})

  }

  station.water_level = water_level
  station.status = getStatus(water_level)

  res.json(station)
})


app.listen(3000, () => {
  console.log("Server running on port 3000");
});
