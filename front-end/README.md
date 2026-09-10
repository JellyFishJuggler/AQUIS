# AQUIS — Frontend (Next.js)

The web frontend of the **AQUIS** groundwater monitoring platform (Aquifer Query and
User Information System). Renders the role-based dashboard UI that will sit on top of
the Express API (`../back-end`) and the ML forecasting surface (see the repo root
[`../Readme.md`](../Readme.md)).

## Stack

| Layer | Choice |
|-------|--------|
| Framework | Next.js 16 (App Router), React 19 |
| Language / styling | TypeScript, Tailwind CSS v4 |
| Charts | Chart.js |

> **Note for agents/editors:** this Next.js version has breaking changes vs older
> training data — read the bundled guides in `node_modules/next/dist/docs/` before
> writing code (see `AGENTS.md`).

## Getting Started

```bash
npm install
npm run dev          # http://localhost:3000 (default)
```

The Express API also listens on `:3000`, so run one of them on a different port if both
are up (e.g. `npm run dev -- -p 3001`).

## Structure

- `app/` — Next.js App Router pages (`app/page.tsx` = dashboard shell)
- `components/dashboard/` — StatGrid, Charts, AlertBanner, QuickActions, StationList
- `components/layout/` — Sidebar, Topbar
- `components/ui`, `components/profile` — shared/profile primitives
- `components` modules are imported via the `@/components` alias

> **Status:** the dashboard is currently a **UI shell** — widget props are hardcoded in
> `app/page.tsx` and there is no API client wired to the backend yet. That wiring is the
> next step (`BACKEND_URL`, `/data/*` and `ML_SERVICE_URL` from the root README).

Full platform setup, ML module and API reference: [`../Readme.md`](../Readme.md) and
[`../docs/api.md`](../docs/api.md).