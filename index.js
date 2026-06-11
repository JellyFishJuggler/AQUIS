// ==============================
// CONFIG
// ==============================
const API = "http://localhost:3000";

// ==============================
// STATE
// ==============================
let allDistricts = [];
let selectedItem = null;
let donutChart = null;
let barChart = null;

// ==============================
// NORMALIZE DATA
// ==============================
function normalize(d) {
  return {
    id: d.id,
    district: d.district,
    state: d.state,
    status: d.status || "SAFE",
    extraction_rate_pct: parseFloat(d.extraction_rate_pct) || 0,
    recharge: parseFloat(d.annual_recharge_ham) || 0,
    extraction: parseFloat(d.total_extraction_ham) || 0,
    net_availability_future_ham: parseFloat(d.net_availability_future_ham) || 0,
  };
}

// ==============================
// INIT
// ==============================
document.addEventListener("DOMContentLoaded", async () => {
  setupQuickActions();

  try {
    const res = await fetch(`${API}/data?t=${Date.now()}`);
    const json = await res.json();

    allDistricts = json.data.map(normalize);

    initCharts();
    render(allDistricts[0]);
    setupModal();

  } catch (err) {
    console.error("FETCH ERROR:", err);
    document.getElementById("stationsContainer").innerHTML =
      `<div class="station-error">⚠ Could not connect to backend.</div>`;
  }
});

// ==============================
// RENDER UI
// ==============================
function render(d) {
  if (!d) return;

  selectedItem = d;

  const hero = document.querySelector(".hero-value");
  if (hero) hero.textContent = d.extraction_rate_pct.toFixed(1) + " %";

  const badge = document.querySelector(".badge");
  if (badge) {
    badge.textContent = d.status;
    badge.style.background =
      d.status === "SAFE" ? "#34e37a" :
        d.status === "OVER_EXPLOITED" ? "#ef4444" :
          d.status === "CRITICAL" ? "#f97316" :
            "#facc15";
    badge.style.color =
      d.status === "SAFE" ? "#065f46" :
        d.status === "OVER_EXPLOITED" ? "#fff" :
          d.status === "CRITICAL" ? "#fff" :
            "#713f12";
  }

  // Update dynamic location target text
  const loc = document.getElementById("locationText");
  if (loc) loc.textContent = d.district;

  // Temperature Simulation
  const tempMetric = document.getElementById("tempMetric");
  if (tempMetric) {
    const charSum = d.district.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const simulatedTemp = 24 + (charSum % 11);
    tempMetric.textContent = `${simulatedTemp}°C`;
  }

  // Power Metric Simulation
  const powerMetric = document.getElementById("powerMetric");
  if (powerMetric) {
    const randomPower = Math.floor(Math.random() * (98 - 78 + 1)) + 78;
    powerMetric.textContent = `${randomPower}%`;

    const powerSub = powerMetric.nextElementSibling;
    if (powerSub) {
      powerSub.textContent = randomPower < 82 ? "Battery Low / Scheduled Maintenance" : "Battery Operational";
      powerSub.style.color = randomPower < 82 ? "#ef4444" : "#404752";
    }
  }

  updateCharts(d);
  loadNearbyStations(d.state, d.district);

  // Dynamic Child-like Message Injector Loop
  const alertBanner = document.getElementById("situationAlertBanner");
  const alertMsg = document.getElementById("bannerMessage");

  if (alertBanner && alertMsg) {
    const currentStatus = d.status ? d.status.toUpperCase() : "SAFE";

    // Reset background classes safely
    alertBanner.className = "situation-banner";

    if (currentStatus === "SAFE") {
      alertBanner.classList.add("safe");
      alertMsg.innerText = "Everything is super safe and perfect right now! Our lovely underground water friends are full, dancing, and having a great time. Let's keep being extra sweet to them and not waste a single drop so they stay happy!";
    }
    else if (currentStatus === "SEMI_CRITICAL") {
      alertBanner.classList.add("semi-critical");
      alertMsg.innerText = "Our water friends down below are feeling a tiny bit tired and thirsty today. They are working extra hard for us! Maybe we can hug them by saving water, closing taps tightly, and helping them refill their little bellies?";
    }
    else {
      // CRITICAL or OVER-EXPLOITED
      alertBanner.classList.add("critical");
      alertMsg.innerText = "Oh no, big emergency! Our poor underground water is almost gone and it's feeling so, so sad and empty. We need to stop hurting it right now! Please, let's be super careful, use only what we really need, and protect it like our best friend!";
    }
  }


}

// ==============================
// NEARBY STATIONS
// ==============================
async function loadNearbyStations(state, currentDistrict) {
  const container = document.getElementById("stationsContainer");
  container.innerHTML = `<div class="station station-skeleton">
    <div class="station-top"><div><strong style="color:#cbd5e1;">Loading…</strong></div></div>
    <div class="progress"><div class="progress-fill" style="width:0%"></div></div>
  </div>`;

  try {
    const res = await fetch(
      `${API}/data?state=${encodeURIComponent(state)}&limit=6&sort_by=extraction_rate_pct&order=desc`
    );
    const json = await res.json();

    const stations = json.data
      .map(normalize)
      .filter(s => s.district !== currentDistrict)
      .slice(0, 4);

    if (stations.length === 0) {
      container.innerHTML = `<div class="station-empty">No other stations found in ${state}.</div>`;
      return;
    }

    container.innerHTML = stations.map((s, i) => buildStationCard(s, i)).join("");

  } catch (err) {
    console.error("STATIONS FETCH ERROR:", err);
    container.innerHTML = `<div class="station-error">⚠ Failed to load nearby stations.</div>`;
  }
}

function buildStationCard(s, index) {
  const rate = s.extraction_rate_pct;
  const barWidth = Math.min(rate, 100).toFixed(1);

  const barColor =
    s.status === "SAFE" ? "#16a34a" :
      s.status === "SEMI_CRITICAL" ? "#facc15" :
        s.status === "CRITICAL" ? "#f97316" :
          "#ef4444";

  const delta = (rate - 70).toFixed(1);
  const deltaLabel = delta >= 0
    ? `<span style="color:#ef4444;">+${delta}% above safe</span>`
    : `<span style="color:#16a34a;">${Math.abs(delta)}% below threshold</span>`;

  const stationId = `${s.state.slice(0, 2).toUpperCase()}-${String(s.id).padStart(3, "0")}`;

  return `
    <div class="station">
      <div class="station-top">
        <div>
          <strong>Station ${stationId}</strong><br>
          <span class="sub">${s.district}, ${s.state}</span>
        </div>
        <div style="text-align:right;">
          <strong>${rate.toFixed(1)}%</strong><br>
          ${deltaLabel}
        </div>
      </div>
      <div class="progress">
        <div class="progress-fill" style="width:${barWidth}%; background:${barColor};"></div>
      </div>
    </div>
  `;
}

// ==============================
// CHARTS MANAGEMENT
// ==============================
function initCharts() {
  const donutEl = document.getElementById("donutChart");
  const barEl = document.getElementById("barChart");
  if (!donutEl || !barEl) return;

  donutChart = new Chart(donutEl, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [50, 50],
        backgroundColor: ["#f97316", "#cbd5e1"],
        borderWidth: 0
      }]
    },
    options: {
      cutout: "70%",
      plugins: { legend: { display: false } }
    }
  });

  barChart = new Chart(barEl, {
    type: "bar",
    data: {
      labels: ["Recharge", "Extraction"],
      datasets: [{
        data: [0, 0],
        backgroundColor: ["#16a34a", "#f97316"],
        borderRadius: 8
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } }
    }
  });
}

function updateCharts(d) {
  if (!donutChart || !barChart) return;

  const pct = d.extraction_rate_pct;

  donutChart.data.datasets[0].data = [Math.min(pct, 100), Math.max(0, 100 - pct)];
  donutChart.update();

  const donutPct = document.getElementById("donutPct");
  if (donutPct) donutPct.textContent = pct.toFixed(1) + "%";

  const legendExtracted = document.getElementById("legendExtracted");
  const legendAvailable = document.getElementById("legendAvailable");
  if (legendExtracted) legendExtracted.textContent = `Extracted (${pct.toFixed(1)}%)`;
  if (legendAvailable) legendAvailable.textContent = `Available (${Math.max(0, 100 - pct).toFixed(1)}%)`;

  barChart.data.datasets[0].data = [d.recharge, d.extraction];
  barChart.update();
}

// ==============================
// MODAL FLOW MANAGEMENT
// ==============================
function setupModal() {
  const modal = document.getElementById("locationModal");
  const locationPill = document.getElementById("locationTrigger"); // Hooked directly to header weather option
  const modalSearch = document.getElementById("modalSearch");

  if (!modal || !locationPill || !modalSearch) return;

  // Clicking the header location pills now fires the modal viewport overlay
  locationPill.addEventListener("click", (e) => {
    modal.style.display = "flex";
    modalSearch.value = ""; // Reset the field inputs
    renderList(allDistricts);
    setTimeout(() => modalSearch.focus(), 50); // Autofocus input field for speed testing
  });

  document.getElementById("cancelBtn").onclick = () => {
    modal.style.display = "none";
  };

  modalSearch.addEventListener("input", (e) => {
    const val = e.target.value.toLowerCase().trim();
    if (!val) { renderList(allDistricts); return; }

    renderList(allDistricts.filter(d =>
      d.district.toLowerCase().includes(val) ||
      d.state.toLowerCase().includes(val)
    ));
  });

  document.getElementById("confirmBtn").onclick = () => {
    if (!selectedItem) return;
    render(selectedItem);
    modal.style.display = "none";
  };
}

function renderList(list) {
  const container = document.getElementById("districtList");
  container.innerHTML = "";

  list.forEach(d => {
    const div = document.createElement("div");
    div.className = "list-item";
    div.innerHTML = `
      <strong>${d.district}</strong>
      <span style="float:right; font-size:12px; color:#64748b;">${d.state}</span>
    `;
    div.onclick = () => {
      document.querySelectorAll(".list-item").forEach(el => el.classList.remove("selected"));
      div.classList.add("selected");
      selectedItem = d;
    };
    container.appendChild(div);
  });
}

// ==============================
// QUICK ACTIONS PROTOTYPING
// ==============================
function setupQuickActions() {
  const anomalyBtn = document.getElementById("actionAnomaly");
  const reportBtn = document.getElementById("actionReport");
  const syncBtn = document.getElementById("actionSync");
  const shareBtn = document.getElementById("actionShare");

  if (anomalyBtn) {
    anomalyBtn.onclick = () => {
      const location = selectedItem ? selectedItem.district : "Current Station";
      alert(`🚨 ANOMALY REPORTED:\nTelemetry mismatch logged for ${location}. Sentinel nodes re-routing verification pings.`);
    };
  }

  if (reportBtn) {
    reportBtn.onclick = () => {
      const location = selectedItem ? `${selectedItem.district} (${selectedItem.state})` : "General";
      alert(`📄 GENERATING REPORT:\nCompiling statistical aquifer volumetric indices for ${location}. Output sent to downloads folder.`);
    };
  }

  if (syncBtn) {
    syncBtn.onclick = () => {
      alert("🔄 FORCING TELEMETRY SYNC:\nFlushing regional grid buffers and re-pooling latest CGWB central register tables...");
    };
  }

  if (shareBtn) {
    shareBtn.onclick = () => {
      if (!selectedItem) return;
      const shareText = `AQUIS Groundwater Data Summary - District: ${selectedItem.district}, State: ${selectedItem.state}, Extraction Velocity Rate: ${selectedItem.extraction_rate_pct.toFixed(1)}% [Status: ${selectedItem.status}]`;

      navigator.clipboard.writeText(shareText)
        .then(() => alert("🔗 LINK COPIED:\nAQUIS secure snapshot summary data exported successfully to clipboard!"))
        .catch(() => alert("Could not write parameters to clipboard."));
    };
  }
}


// Live current date display engine
const dateOptions = { weekday: 'long', month: 'short', day: 'numeric' };
const today = new Date().toLocaleDateString('en-US', dateOptions);
document.getElementById('liveDashboardDate').innerText = today;
