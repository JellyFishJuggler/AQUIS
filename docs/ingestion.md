# AQUIS Ingestion

## Telemetry Ingestion

### Source
NWDP API: `https://nwdp.nwic.gov.in/api/3/action/datastore_search`
Resource ID: `84bfda45-8ead-436d-9c8e-f7a93ee57522`
Total records: ~7.4 million

### Process
1. Fetch pages from NWDP API (100 records per page)
2. For each record:
   - Parse and clean all text fields (trim, handle `-`, `NA`, `N/A`)
   - Parse timestamps (DD-MM-YYYY HH:MM format)
   - Parse numeric values (groundwater level, coordinates)
   - Upsert station metadata (deduplicate by external_station_id)
   - Insert observation with unique constraint (station_id, observed_at)
3. Track in ingestion_runs table
4. Skip duplicates silently (ON CONFLICT DO NOTHING)

### Features
- Paginated fetching with offset
- Retry with exponential backoff (3 attempts)
- 30-second request timeout
- Batch processing (100 records per batch)
- Deduplication via unique constraint
- Ingestion run tracking
- Error logging with samples

### Commands
```bash
# Full ingestion
npm run import-telemetry

# Limited ingestion (first 1000 records)
npm run import-telemetry -- --limit 1000

# Custom batch size
npm run import-telemetry -- --batch 200
```

### API Trigger
```bash
curl -X POST http://localhost:3000/ingestion/telemetry \
  -H "Content-Type: application/json" \
  -d '{"batchSize": 100, "limit": 5000}'
```

## Assessment Ingestion

### Source
Excel files in `back-end/db/centralreport_ingres/`

### Available Data
| File | Year | States | Records |
|------|------|--------|---------|
| CentralReport...016614 | 2025-2026 | 2 (partial) | 75 |
| CentralReport...130368 | 2024-2025 | 37 | 736 |
| CentralReport...82365 | 2025-2026 | 9 (partial) | 183 |
| CentralReport...454607 | 2024-2025 | 37 | 736 |
| CentralReport...541485 | 2023-2024 | 37 | 724 |
| CentralReport...580743 | 2022-2023 | 37 | 714 |
| CentralReport...712104 | 2021-2022 | 35 | 706 |
| CentralReport...745929 | 2019-2020 | 32 | 652 |
| CentralReport...770926 | 2016-2017 | 29 | 595 |

### Process
1. Read each Excel file (sheet: "GEC")
2. Parse 3-level merged headers (154 columns)
3. Detect assessment year from data content
4. Extract key metrics (recharge, extraction, classification)
5. Upsert assessment units (deduplicate)
6. Insert/update assessment records (ON CONFLICT DO UPDATE)
7. Store raw_data as JSONB for auditability
8. Track in ingestion_runs

### Cleaning Rules
- Trim all text fields
- Convert `-`, `NA`, `N/A`, empty strings to NULL
- Parse numeric values with parseFloat
- Validate: no negative extraction, no negative rainfall
- Derive extraction rate and classification from raw values

### Idempotency
- Re-importing the same file updates existing records
- No duplicate unit+year combinations
- Historical years never overwritten

### Commands
```bash
# Import all Excel files
npm run import-assessments

# Import with forced year
npm run import-assessments -- --year 2024-2025

# Import from custom directory
npm run import-assessments -- --dir /path/to/files
```
