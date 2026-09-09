"""08_lulc — ISRO Bhuvan LULC (50K) district statistics for the 29 UP districts.

Fetches per-district Land Use / Land Cover areas (24 classes, l01-l24) from
Bhuvan's Thematic Statistics API for the two released 50K cycles (2005-06 and
2011-12), turns them into %.of-district static features, and correlates them
with observed district GWL summary stats.

API   : https://bhuvan-app1.nrsc.gov.in/api/lulc/curljson.php
Params: year=0506|1112, distcode=<4-digit: state(2)+district(2)>, token=<key>
Header: Content-Type: application/x-www-form-urlencoded  (required, even for GET)
Key   : read from <repo>/.env -> LULC_STATISTICS_API_KEY (gitignored)
Outputs:
  data/meta/lulc_long.csv            long: district, year, class, area_sqkm, pct
  data/soil/lulc_district.csv        wide: District + 9 static LULC pct features (2011-12)
  outputs/correlation_lulc.csv       district LULC% x district GWL summary correlations
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent  # AQUIS/ -> holds .env
META_DIR = ROOT / "data" / "meta"
OUT_DIR = ROOT / "outputs"

API = "https://bhuvan-app1.nrsc.gov.in/api/lulc/curljson.php"
YEARS = {"1112": "2011-12", "0506": "2005-06"}
DISTRICT_CODES = {  # Bhuvan (name -> 4-digit code), normalized key
    "Saharanpur": "0901", "Muzaffarnagar": "0902", "Bijnor": "0903", "Moradabad": "0904",
    "Rampur": "0905", "Jyotiba Phule Nagar": "0906", "Meerut": "0907", "Baghpat": "0908",
    "Ghaziabad": "0909", "Gautam Buddha Nagar": "0910", "Bulandshahr": "0911",
    "Aligarh": "0912", "Hathras": "0913", "Mathura": "0914", "Agra": "0915",
    "Firozabad": "0916", "Etah": "0917", "Mainpuri": "0918", "Badaun": "0919",
    "Bareilly": "0920", "Pilibhit": "0921", "Shahjahanpur": "0922", "Lakhimpur Kheri": "0923",
    "Sitapur": "0924", "Hardoi": "0925", "Unnao": "0926", "Lucknow": "0927", "Rae Bareli": "0928",
    "Farrukhabad": "0929", "Kannauj": "0930", "Etawah": "0931", "Auraiya": "0932",
    "Kanpur Dehat": "0933", "Kanpur": "0934", "Jalaun": "0935", "Jhansi": "0936",
    "Lalitpur": "0937", "Hamirpur": "0938", "Mahoba": "0939", "Banda": "0940",
    "Chitrakoot": "0941", "Fatehpur": "0942", "Pratapgarh": "0943", "Kaushambi": "0944",
    "Allahabad": "0945", "Bara Banki": "0946", "Faizabad": "0947", "Ambedkar Nagar": "0948",
    "Sultanpur": "0949", "Bahraich": "0950", "Shravasti": "0951", "Balrampur": "0952",
    "Gonda": "0953", "Siddharth Nagar": "0954", "Basti": "0955", "Sant Kabir Nagar": "0956",
    "Maharajganj": "0957", "Gorakhpur": "0958", "Kushinagar": "0959", "Deoria": "0960",
    "Azamgarh": "0971", "Mau": "0962", "Ballia": "0963", "Jaunpur": "0964",
    "Ghazipur": "0965", "Chandauli": "0966", "Varanasi": "0967", "Sant Ravi Das Nagar": "0968",
    "Mirzapur": "0969", "Sonbhadra": "0970", "Kansiramnagar": "0972",
}
LULC_CLASSES = {  # l-code -> class label (as documented by Bhuvan getlucode)
    "l01": "Builtup,Urban", "l02": "Builtup,Rural", "l03": "Builtup,Mining",
    "l04": "Agriculture,Crop land", "l05": "Agriculture,Plantation",
    "l06": "Agriculture,Fallow", "l07": "Agriculture, Shifting Cultivation",
    "l08": "Forest,Evergreen/Semi evergreen", "l09": "Forest,Deciduous",
    "l10": "Forest,Forest Plantation", "l11": "Forest,Scrub Forest",
    "l12": "Forest,Swamp/Mangroves", "l13": "Grass/Grazing",
    "l14": "Barren/Wastelands, Salt Affected", "l15": "Barren/Wastelands, Gullied/Ravinus",
    "l16": "Barren/Wastelands, Scrub land", "l17": "Barren/Wastelands, Sandy area",
    "l18": "Barren/Wastelands, Barren rocky", "l19": "Barren/Wastelands, Rann",
    "l20": "Wetlands/Water, Inland Wetland", "l21": "Wetlands/Water, Coastal Wetland",
    "l22": "Wetlands/Water, River/Stream/Canal", "l23": "Wetlands/Water, Reservoir/Lake/Pond",
    "l24": "Snow and Glacier",
}
# static per-district feature groups (percent of district total area, 2011-12 cycle)
LULC_GROUPS = {
    "lulc_builtup_pct": ["l01", "l02", "l03"],
    "lulc_cropland_pct": ["l04"],
    "lulc_plantation_pct": ["l05"],
    "lulc_fallow_pct": ["l06"],
    "lulc_shifting_cult_pct": ["l07"],
    "lulc_forest_pct": ["l08", "l09", "l10", "l11", "l12"],
    "lulc_grass_pct": ["l13"],
    "lulc_wasteland_pct": ["l14", "l15", "l16", "l17", "l18", "l19"],
    "lulc_water_pct": ["l20", "l21", "l22", "l23"],
}
LULC_COLS = list(LULC_GROUPS)

CODE_MAP = {name.upper().replace(" ", ""): code for name, code in DISTRICT_CODES.items()}
CODE_MAP["BUDAUN"] = "0919"       # repo spelling; Bhuvan lists "Badaun"
CODE_MAP["SHRAWASTI"] = "0951"    # repo spelling; Bhuvan lists "Shravasti"


def read_env(repo: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    f = repo / ".env"
    if not f.exists():
        return env
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip("'\"")
    return env


def norm(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def fetch_district(name: str, code: str, year: str, token: str) -> dict[str, float | str]:
    params = urllib.parse.urlencode({"year": year, "distcode": code, "token": token})
    req = urllib.request.Request(
        f"{API}?{params}", headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    total = float(data.get("totalarea") or data.get("1"))
    out: dict[str, float | str] = {"District": name, "total_area_sqkm": total}
    for i in range(1, 25):
        key = f"l{i:02d}"
        v = data.get(key) or data.get(str(i + 1))
        out[f"l{i:02d}_sqkm"] = float(v) if v is not None else np.nan
    return out


def main() -> None:
    env = read_env(REPO)
    token = env.get("LULC_STATISTICS_API_KEY", "").strip()
    if not token:
        print("MISSING LULC_STATISTICS_API_KEY in <repo>/.env", flush=True)
        return

    districts = sorted(json.loads((META_DIR / "selected_districts.json").read_text()))
    code_of = {norm(d): CODE_MAP.get(norm(d)) for d in districts}
    missing = {d: c for d, c in code_of.items() if not c}
    if missing:
        print(f"no Bhuvan code for: {list(missing)}", flush=True)
        return

    META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict] = []
    wide_rows: list[dict] = []
    n = 0
    for i, d in enumerate(districts, 1):
        code = code_of[norm(d)]
        for year, label in YEARS.items():
            row = fetch_district(d, code, year, token)
            total = float(row["total_area_sqkm"])
            for code_l, classname in LULC_CLASSES.items():
                a = float(row[f"{code_l}_sqkm"])
                long_rows.append({"district": d, "year": label, "class": classname,
                                  "area_sqkm": a, "pct": 100.0 * a / total if total else np.nan})
            if year == "1112":
                wide = {"District": d}
                total = max(total, 1e-9)
                for group, codes in LULC_GROUPS.items():
                    wide[group] = 100.0 * sum(float(row[f"{c}_sqkm"]) for c in codes) / total
                wide_rows.append(wide)
            n += 1
            print(f"[{n}] {d}: total {total:.0f} km2 ({label})", flush=True)
            time.sleep(0.3)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(META_DIR / "lulc_long.csv", index=False)
    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv(ROOT / "data" / "soil" / "lulc_district.csv", index=False)
    print(f"lulc_long: {len(long_df):,} rows; lulc_district: {len(wide_df)} districts",
          flush=True)

    # ---- district GWL summaries + LULC x GWL correlation -----------------------
    tbl = pd.read_parquet(ROOT / "data" / "aligned" / "table_6h.parquet")
    tbl["date"] = pd.to_datetime(tbl["time"])
    g = tbl.dropna(subset=["gwl"]).groupby("District")["gwl"]
    summ = pd.DataFrame({
        "District": [k for k, _ in g],
        "gwl_mean_m": [v.mean() for _, v in g],
        "gwl_median_m": [v.median() for _, v in g],
        "gwl_std_m": [v.std() for _, v in g],
        "gwl_amp85_m": [np.nanpercentile(v, 95) - np.nanpercentile(v, 5) for _, v in g],
    })
    corr = wide_df.merge(summ, on="District", how="inner")
    rows = []
    for c in LULC_COLS:
        for m in ["gwl_mean_m", "gwl_median_m", "gwl_std_m", "gwl_amp85_m"]:
            x, y = corr[c].astype(float), corr[m].astype(float)
            mask = x.notna() & y.notna()
            r = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() > 8 else np.nan
            rows.append({"lulc_feature": c, "gwl_metric": m,
                         "districts": int(mask.sum()), "pearson_r": r})
    corr_df = pd.DataFrame(rows).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
    corr_df.to_csv(OUT_DIR / "correlation_lulc.csv", index=False)
    print("correlation_lulc ->", OUT_DIR / "correlation_lulc.csv", flush=True)

    meta = {
        "source": "ISRO Bhuvan LULC 50K Thematic Statistics",
        "api": API,
        "cycles": list(YEARS.values()),
        "classes": len(LULC_CLASSES),
        "features_used": "2011-12 cycle (latest 50K); percent of district total area",
        "districts": len(wide_df),
        "key_used": "LULC_STATISTICS_API_KEY (<repo>/.env)",
        "queried_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (META_DIR / "lulc_meta.json").write_text(json.dumps(meta, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()