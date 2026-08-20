const path = require("path");
const fs = require("fs");
const XLSX = require("xlsx");
const { classifyStatus, computeExtractionRate } = require("./groundwaterClassification");
const ingestionRunService = require("./ingestionRunService");
const assessmentService = require("./assessmentService");
const { query } = require("../db/pool");

const CENTRALREPORT_DIR = path.join(__dirname, "..", "db", "centralreport_ingres");

function cleanText(val) {
  if (val == null) return null;
  const s = String(val).trim();
  if (s === "" || s === "-" || s.toLowerCase() === "na" || s.toLowerCase() === "n/a" || s.toLowerCase() === "null") return null;
  return s;
}

function parseNumeric(val) {
  if (val == null) return null;
  const s = String(val).trim();
  if (s === "" || s === "-" || s.toLowerCase() === "na" || s.toLowerCase() === "n/a") return null;
  const num = parseFloat(s);
  return isNaN(num) ? null : num;
}

function detectAssessmentYear(workbook) {
  const sheetNames = workbook.SheetNames || [];
  for (const name of sheetNames) {
    const match = name.match(/(\d{4})[-_](\d{4})/);
    if (match) return `${match[1]}-${match[2]}`;
  }
  return null;
}

function detectYearFromData(worksheet) {
  const range = XLSX.utils.decode_range(worksheet["!ref"] || "A1");
  for (let r = 0; r <= Math.min(range.e.r, 20); r++) {
    for (let c = range.s.c; c <= range.e.c; c++) {
      const cell = worksheet[XLSX.utils.encode_cell({ r, c })];
      if (cell && cell.v) {
        const val = String(cell.v);
        const match = val.match(/(\d{4})[-–](\d{4})/);
        if (match) return `${match[1]}-${match[2]}`;
      }
    }
  }
  return null;
}

function getAssessmentYear(filePath, workbook) {
  const fromName = detectAssessmentYear(workbook);
  if (fromName) return fromName;
  const fromData = detectYearFromData(workbook.Sheets[workbook.SheetNames[0]]);
  if (fromData) return fromData;
  const basename = path.basename(filePath);
  const tsMatch = basename.match(/(\d{13})/);
  if (tsMatch) {
    const ts = parseInt(tsMatch[1], 10);
    const d = new Date(ts);
    const year = d.getFullYear();
    return `${year - 1}-${year}`;
  }
  return null;
}

function parseExcelRow(row, headers) {
  const get = (key) => {
    const idx = headers.indexOf(key);
    return idx >= 0 ? row[idx] : undefined;
  };
  const getN = (key) => parseNumeric(get(key));
  const getText = (key) => cleanText(get(key));

  const state = getText("STATE");
  const district = getText("DISTRICT");
  const assessmentUnit = getText("ASSESSMENT UNIT");

  if (!state || !district) return null;

  const rainfall = getN("Rainfall (mm)_Total");
  const rechargeWorthyArea = getN("Recharge Worthy Area (ha)_Total");
  const rechargeRainfall = getN("Ground Water Recharge_Rainfall Recharge_Total");
  const rechargeCanals = getN("Ground Water Recharge_Canals_Total");
  const rechargeSurfaceIrr = getN("Ground Water Recharge_Surface Water Irrigation_Total");
  const rechargeGwIrr = getN("Ground Water Recharge_Ground Water Irrigation_Total");
  const rechargeTanks = getN("Ground Water Recharge_Tanks and Ponds_Total");
  const rechargeWcs = getN("Ground Water Recharge_Water Conservation Structure_Total");
  const rechargePipelines = getN("Ground Water Recharge_Pipelines_Total");
  const rechargeSewage = getN("Ground Water Recharge_Sewages and Flash Flood Channels_Total");
  const totalRecharge = getN("Ground Water Recharge_Total GW Recharge_Total");
  const streamRecharge = getN("Inflows and Outflows_Stream Recharges_Total");
  const annualRecharge = getN("Annual Ground water Recharge (ham)_Total");
  const envFlows = getN("Environmental Flows (ham)_Total");
  const annualExtractable = getN("Annual Extractable GW Resource (ham)_Total");
  const extractionDomestic = getN("GW Extraction_Domestic_Total");
  const extractionIndustrial = getN("GW Extraction_Industrial_Total");
  const extractionIrrigation = getN("GW Extraction_Irrigation_Total");
  const totalExtraction = getN("GW Extraction_Total_Total");
  const stageOfExtraction = getN("Stage of GW Extraction (%)_Total");
  const allocationDomestic = getN("Allocation for Domestic (projected 2025)_Total");
  const netAvailability = getN("Net Annual GW Availability Future Use_Total");
  const qualityMajor = getText("Quality Tagging_Major Parameter Present_C");
  const qualityOther = getText("Quality Tagging_Other Parameters Present_C");
  const coastalArea = getN("Coastal Areas_Total");
  const storageUnconfinedFresh = getN("In-Storage Unconfined GW Resources_Fresh");
  const storageUnconfinedSaline = getN("In-Storage Unconfined GW Resources_Saline");
  const availUnconfinedFresh = getN("Total GW Availability Unconfined_Fresh");
  const availUnconfinedSaline = getN("Total GW Availability Unconfined_Saline");
  const dynamicConfinedFresh = getN("Dynamic Confined GW Resources_Fresh");
  const dynamicConfinedSaline = getN("Dynamic Confined GW Resources_Saline");
  const storageConfinedFresh = getN("In-Storage Confined GW Resources_Fresh");
  const storageConfinedSaline = getN("In-Storage Confined GW Resources_Saline");
  const totalConfinedFresh = getN("Total Confined GW Resources_Fresh");
  const totalConfinedSaline = getN("Total Confined GW Resources_Saline");
  const dynamicSemiFresh = getN("Dynamic Semi-Confined GW Resources_Fresh");
  const dynamicSemiSaline = getN("Dynamic Semi-Confined GW Resources_Saline");
  const storageSemiFresh = getN("In-Storage Semi-Confined GW Resources_Fresh");
  const storageSemiSaline = getN("In-Storage Semi-Confined GW Resources_Saline");
  const totalSemiFresh = getN("Total Semi-Confined GW Resources_Fresh");
  const totalSemiSaline = getN("Total Semi-Confined GW Resources_Saline");
  const totalGwFresh = getN("Total GW Availability in Area_Fresh");
  const totalGwSaline = getN("Total GW Availability in Area_Saline");
  const totalGeoArea = getN("Total Geographical Area (ha)_Total");
  const hillyArea = getN("Total Geographical Area (ha)_Hilly Area");

  let extractionRate = stageOfExtraction;
  if (extractionRate == null && totalExtraction != null && annualExtractable != null) {
    extractionRate = computeExtractionRate(totalExtraction, annualExtractable);
  }
  const status = classifyStatus(extractionRate);

  return {
    state, district, assessment_unit: assessmentUnit,
    rainfall_mm: rainfall,
    recharge_worthy_area_ha: rechargeWorthyArea,
    recharge_rainfall_ham: rechargeRainfall,
    recharge_canals_ham: rechargeCanals,
    recharge_surface_irr_ham: rechargeSurfaceIrr,
    recharge_gw_irr_ham: rechargeGwIrr,
    recharge_tanks_ponds_ham: rechargeTanks,
    recharge_wcs_ham: rechargeWcs,
    recharge_pipelines_ham: rechargePipelines,
    recharge_sewage_ff_ham: rechargeSewage,
    total_recharge_ham: totalRecharge,
    stream_recharge_ham: streamRecharge,
    annual_recharge_ham: annualRecharge,
    environmental_flows_ham: envFlows,
    annual_extractable_gw_ham: annualExtractable,
    extraction_domestic_ham: extractionDomestic,
    extraction_industrial_ham: extractionIndustrial,
    extraction_irrigation_ham: extractionIrrigation,
    total_extraction_ham: totalExtraction,
    stage_of_extraction_pct: stageOfExtraction,
    extraction_rate_pct: extractionRate,
    net_availability_future_ham: netAvailability,
    allocation_domestic_ham: allocationDomestic,
    quality_tagging_major: qualityMajor,
    quality_tagging_other: qualityOther,
    coastal_area_ham: coastalArea,
    storage_unconfined_fresh: storageUnconfinedFresh,
    storage_unconfined_saline: storageUnconfinedSaline,
    availability_unconfined_fresh: availUnconfinedFresh,
    availability_unconfined_saline: availUnconfinedSaline,
    dynamic_confined_fresh: dynamicConfinedFresh,
    dynamic_confined_saline: dynamicConfinedSaline,
    storage_confined_fresh: storageConfinedFresh,
    storage_confined_saline: storageConfinedSaline,
    total_confined_fresh: totalConfinedFresh,
    total_confined_saline: totalConfinedSaline,
    dynamic_semiconfined_fresh: dynamicSemiFresh,
    dynamic_semiconfined_saline: dynamicSemiSaline,
    storage_semiconfined_fresh: storageSemiFresh,
    storage_semiconfined_saline: storageSemiSaline,
    total_semiconfined_fresh: totalSemiFresh,
    total_semiconfined_saline: totalSemiSaline,
    total_gw_availability_fresh: totalGwFresh,
    total_gw_availability_saline: totalGwSaline,
    total_gw_availability_ham: (totalGwFresh != null && totalGwSaline != null) ? totalGwFresh + totalGwSaline : totalGwFresh || totalGwSaline || null,
    total_geographical_area_ha: totalGeoArea,
    hilly_area_ha: hillyArea,
    status,
  };
}

function buildColumnMap(worksheet) {
  const range = XLSX.utils.decode_range(worksheet["!ref"] || "A1");
  const merges = worksheet["!merges"] || [];

  const headerRows = [];
  for (let r = 0; r <= Math.min(range.e.r, 15); r++) {
    const rowVals = [];
    for (let c = range.s.c; c <= range.e.c; c++) {
      const cell = worksheet[XLSX.utils.encode_cell({ r, c })];
      rowVals.push(cell ? String(cell.v || "").trim() : "");
    }
    headerRows.push(rowVals);
  }

  const colHeaders = [];
  for (let c = range.s.c; c <= range.e.c; c++) {
    const parts = [];
    for (let r = 0; r < Math.min(headerRows.length, 10); r++) {
      let val = headerRows[r][c];
      if (!val) {
        for (const merge of merges) {
          if (r >= merge.s.r && r <= merge.e.r && c >= merge.s.c && c <= merge.e.c) {
            val = headerRows[merge.s.r][merge.s.c];
            break;
          }
        }
      }
      if (val && val !== "C" && val !== "NC" && val !== "PQ") {
        parts.push(val);
      }
    }
    colHeaders.push(parts.join("_"));
  }
  return colHeaders;
}

function parseExcelFile(filePath) {
  console.log(`  Reading ${path.basename(filePath)}...`);
  const workbook = XLSX.readFile(filePath, { type: "file" });
  const sheetName = workbook.SheetNames[0];
  if (!sheetName) return { year: null, records: [], errors: [] };

  const worksheet = workbook.Sheets[sheetName];
  const year = getAssessmentYear(filePath, workbook);
  const headers = buildColumnMap(worksheet);

  const range = XLSX.utils.decode_range(worksheet["!ref"] || "A1");
  const records = [];
  const errors = [];

  let dataStartRow = 0;
  for (let r = 0; r <= Math.min(range.e.r, 20); r++) {
    const hasState = headers.some(h => h.includes("STATE"));
    if (hasState) {
      const cellVal = worksheet[XLSX.utils.encode_cell({ r, c: 1 })];
      if (cellVal && cellVal.v && String(cellVal.v).trim().length > 0) {
        dataStartRow = r;
        break;
      }
    }
    dataStartRow = r + 1;
  }

  for (let r = dataStartRow; r <= range.e.r; r++) {
    const rowData = [];
    for (let c = range.s.c; c <= range.e.c; c++) {
      const cell = worksheet[XLSX.utils.encode_cell({ r, c })];
      rowData.push(cell ? cell.v : null);
    }
    const firstNonEmpty = rowData.find(v => v != null && String(v).trim() !== "");
    if (!firstNonEmpty) continue;

    try {
      const parsed = parseExcelRow(rowData, headers);
      if (parsed) {
        parsed.raw_data = {};
        headers.forEach((h, i) => {
          if (h && rowData[i] != null) parsed.raw_data[h] = rowData[i];
        });
        records.push(parsed);
      } else {
        errors.push({ row: r, reason: "missing state/district" });
      }
    } catch (err) {
      errors.push({ row: r, reason: err.message });
    }
  }

  return { year, records, errors };
}

async function ingestAll({ sourceDir = CENTRALREPORT_DIR, assessmentYear = null } = {}) {
  const files = fs.readdirSync(sourceDir).filter(f => f.endsWith(".xlsx"));
  if (files.length === 0) {
    console.log("No Excel files found in", sourceDir);
    return { runs: [], totalRecords: 0 };
  }

  console.log(`\n📊 Assessment ingestion: ${files.length} files found`);
  const allResults = [];

  for (const file of files) {
    const filePath = path.join(sourceDir, file);
    try {
      const { year, records, errors } = parseExcelFile(filePath);
      const effectiveYear = assessmentYear || year;

      if (!effectiveYear) {
        console.log(`  ⚠️  Skipping ${file}: cannot determine assessment year`);
        allResults.push({ file, status: "skipped", reason: "no year" });
        continue;
      }

      const runId = await ingestionRunService.createRun("assessment_excel", file, {
        year: effectiveYear, totalRecords: records.length, parseErrors: errors.length,
      });

      let inserted = 0, updated = 0, rejected = 0, duplicates = 0;

      for (const record of records) {
        try {
          const unitId = await assessmentService.upsertUnit(record.state, record.district, record.assessment_unit);
          const recordId = await assessmentService.insertRecord({
            ...record,
            assessment_unit_id: unitId,
            assessment_year: effectiveYear,
            source_file: file,
            source_record_id: `${file}_${record.state}_${record.district}_${record.assessment_unit || ""}`,
            ingestion_run_id: runId,
          });
          if (recordId) inserted++;
          else duplicates++;
        } catch (err) {
          rejected++;
          if (rejected <= 10) console.log(`    ❌ Row rejected: ${err.message}`);
        }
      }

      await ingestionRunService.finishRun(runId, {
        status: "completed",
        recordsSeen: records.length,
        recordsInserted: inserted,
        recordsDuplicates: duplicates,
        recordsRejected: rejected + errors.length,
        errorCount: errors.length,
        errorSummary: errors.length > 0 ? { samples: errors.slice(0, 10) } : null,
      });

      console.log(`  ✅ ${file} (${effectiveYear}): ${inserted} inserted, ${duplicates} duplicates, ${rejected + errors.length} rejected`);
      allResults.push({ file, year: effectiveYear, inserted, duplicates, rejected: rejected + errors.length, runId });
    } catch (err) {
      console.error(`  ❌ Failed to process ${file}: ${err.message}`);
      allResults.push({ file, status: "failed", error: err.message });
    }
  }

  const totalRecords = allResults.reduce((sum, r) => sum + (r.inserted || 0), 0);
  console.log(`\n📊 Assessment ingestion complete: ${totalRecords} total records inserted across ${allResults.length} files`);
  return { runs: allResults, totalRecords };
}

module.exports = { ingestAll, parseExcelFile, buildColumnMap, detectAssessmentYear };
