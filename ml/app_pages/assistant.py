"""Assistant page — station-locked natural-language Q&A over live AQUIS data.

Facts (latest level, changes, outliers, 30-day XGBoost outlook) are computed
deterministically from ``table_6h`` + the pooled 30-day model; llama3.2:3b via
Ollama (local) only phrases the answer. Works without Ollama: the instant data
panel above the chat always renders.
"""

import pandas as pd
import streamlit as st

from _assistant import (
    StationAssistant,
    get_df,
    ollama_model,
    ollama_status,
    district_names,
    station_names,
)


def _fmt_dr(x):
    return "n/a" if x is None else f"{'+' if x >= 0 else ''}{x:.3f}"

st.set_page_config(page_title="AQUIS — assistant", page_icon=":material/smart_toy:", layout="wide")
st.title("Assistant — groundwater data in plain language")

df = get_df()
status = ollama_status()
model_name = status["model"]
up = status["server"] and model_name in status["models"]

if not status["server"]:
    st.warning(
        "Ollama server not reachable. Start it with `ollama serve` (and ensure the "
        f"model is pulled). **The instant data panel below still works** — chat "
        "answers need the local model."
    )
elif not up:
    st.info(f"Ollama reachable, but model **`{model_name}`** is not pulled. Run:")
    st.code(f"ollama pull {model_name}", language="bash")

with st.expander("Model status", icon=":material/dns:"):
    shown = {k: (sorted(v) if isinstance(v, set) else v) for k, v in status.items()}
    st.json(shown)
    st.caption("Re-run the page (or press R) to re-check after starting Ollama / pulling a model.")

recency = df.groupby("Station")["time"].max().sort_values(ascending=False)
stations = list(recency.index.astype(str))

col1, col2 = st.columns([2, 1])
with col1:
    station = st.selectbox("Station", stations, key="as_station")
with col2:
    st.caption(f"{len(stations)} selected stations — most recently updated first.")

assistant = StationAssistant()
facts = assistant.facts(station)

m1, m2, m3, m4, m5 = st.columns(5)
if facts:
    m1.metric("Latest level", f"{facts['last']:.2f} m", f"on {facts['last_date']}", border=True)
    m2.metric("30-day change", _fmt_dr(facts.get("change_30d")), border=True)
    m3.metric("7-day change", _fmt_dr(facts.get("change_7d")), border=True)
    m4.metric("Obs. range", f"{facts['min']:.2f}–{facts['max']:.2f} m", border=True)
    m5.metric("Samples", f"{facts['n_obs']:,}", border=True)
    st.caption(
        f"District **{facts.get('district')}** median "
        f"{facts.get('district_median'):.2f} m across {facts.get('district_n_stations')} "
        "stations. Rising = water level went up (GWL is a level in metres). "
        "Rising/falling names: rising level = value up, declining = value down."
    )
    f = facts.get("forecast")
    if f and f.get("plausible"):
        st.success(
            f"**30-day outlook (XGBoost): {f['day30_pred']:.2f} m** "
            f"({f['change_30d_pred']:+.2f} m change, ±{f['band_half']:.1f} m band "
            f"around a {f['anchor']:.2f} m anchor)."
        )
    elif f and not f.get("plausible"):
        st.warning("Forecast model unstable for this station — observed trend only.")
else:
    st.info("No data for that selection.")

if facts:
    tcol1, tcol2 = st.columns(2)
    drivers = facts.get("drivers") or []
    annual = facts.get("annual") or []
    rain = facts.get("rain_recent") or {}
    dc = facts.get("district_context") or {}
    with tcol1:
        with st.expander("Factors driving this station", icon=":material/science:"):
            if drivers:
                dtab = pd.DataFrame(drivers)[["driver", "corr", "p", "n"]]
                dtab = dtab.rename(
                    columns={"driver": "Factor", "corr": "Spearman r",
                             "p": "p-value", "n": "pairs"})
                st.dataframe(
                    dtab, hide_index=True, use_container_width=True,
                    column_config={
                        "Spearman r": st.column_config.NumberColumn(format="%+.3f"),
                        "p-value": st.column_config.NumberColumn(format="%.4f"),
                    })
                st.caption(
                    "r = correlation between the factor and the station's GWL over its "
                    "6h history (Spearman). The largest |r| tells you which factor "
                    "moves the level the most here."
                )
            else:
                st.caption("Not enough co-observed driver data for this station.")
            if rain.get("rain_7d") is not None or rain.get("rain_30d") is not None:
                st.markdown(
                    f"**Recent rain**: 7d `{_fmt_dr(rain.get('rain_7d'))}`, "
                    f"30d `{_fmt_dr(rain.get('rain_30d'))}`, "
                    f"90d `{_fmt_dr(rain.get('rain_90d'))}` mm (last "
                    f"{rain.get('last_rain_date')})."
                )
            if annual:
                atab = pd.DataFrame(annual)[["year", "mean", "min", "max", "rain_mm", "n"]]
                atab = atab.rename(
                    columns={"year": "Year", "mean": "Mean (m)", "min": "Min (m)",
                             "max": "Max (m)", "rain_mm": "Rain (mm)", "n": "n"})
                st.dataframe(atab, hide_index=True, use_container_width=True)
                st.caption("Mean/min/max GWL and total rain per calendar year — "
                           "compare last 1-2 years to see which factor moved the level.")
    with tcol2:
        with st.expander("Precautions & strategy", icon=":material/health_and_safety:"):
            prec = facts.get("precautions") or []
            if prec:
                for p in prec:
                    icon = {
                        "info": ":material/lightbulb:",
                        "watch": ":material/warning:",
                        "action": ":material/error:",
                    }.get(p["level"], ":material/lightbulb:")
                    level = {"info": "🔵", "watch": "🟡", "action": "🔴"}.get(p["level"], p["level"])
                    st.markdown(f"**{level} {p['title']}**")
                    st.markdown(p["why"])
                    st.divider()
                st.caption("Deterministic rule-based advisories. Ask the assistant "
                           "e.g. “precautions for this station?” to get them phrased "
                           "as concrete suggestions.")
            else:
                st.caption("No active advisories for this station right now.")
            if dc.get("n_analysed"):
                st.caption(
                    f"District context: median **{facts.get('district_median'):.2f} m** "
                    f"over {dc['n_analysed']} stations in {facts.get('district')}."
                )

st.markdown("---")

if st.session_state.get("as_pinned_station") != station:
    st.session_state.as_pinned_station = station
    district = facts.get("district") if facts else ""
    st.session_state.as_messages = [
        {"role": "assistant",
         "content": f"Hello! I can answer about the pinned station **{station}** "
                    f"(district {district}) or any station/district you name — e.g. "
                    "“latest level”, “trend over 30 days”, “what's driving the level?”, "
                    "“rain effect last 1-2 years?”, or “suggest precautions?”"}
    ]

for msg in st.session_state.as_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(f"Ask about {station}"):
    history = st.session_state.as_messages[-6:]
    st.session_state.as_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not up:
            st.markdown(
                f"⚠️ **Ollama not available.** Start it with `ollama serve` and ensure "
                f"`{model_name}` is pulled — then ask again."
            )
            answer = "⚠️ Ollama not available — no answer generated."
        else:
            with st.spinner("Querying AQUIS data + generating answer (local model, ~15-60s)..."):
                try:
                    res = assistant.answer(prompt, station, history=history)
                    answer = res.get("answer", "Sorry, could not generate an answer.")
                except Exception as e:  # noqa: BLE001 - surface setup issues, don't crash
                    answer = (
                        f"⚠️ **Could not reach the assistant.** Is the Ollama server "
                        f"running with `{model_name}` pulled? Error: `{e}`"
                    )
            st.markdown(answer)
    st.session_state.as_messages.append({"role": "assistant", "content": answer})

with st.expander("About this assistant", icon=":material/info:"):
    st.markdown(
        "**Stack**: LangChain + Ollama (local, open-source) — nothing leaves this machine.\n\n"
        f"- **Model**: `{model_name}` (overridable with `AQUIS_OLLAMA_MODEL` in the repo `.env`)\n"
        "- **Data**: live `data/aligned/table_6h.parquet` (observed trends, latest levels)\n"
        "- **Forecast**: the app's pooled XGBoost 30-day change model (same as the Forecast page)\n"
        "- **Factors**: Spearman correlations of rain/temp/river/canal/etc. vs GWL, per-station rain totals, and per-year annual trends (last 1-2 yrs)\n"
        "- **Precautions**: deterministic rule-based advisories (decline, drawdown, weak recharge, forecast uncertainty)\n"
        "- **Scope**: answers about the pinned station "
        f"(**{station}**), plus any station/district you mention in the question "
        f"({len(station_names())} stations / {len(district_names())} districts accessible)\n"
        "- **Convention**: GWL is a water-table level in metres — rising level = value up\n\n"
        "The LLM only phrases an answer from pre-computed facts (no made-up numbers)."
    )