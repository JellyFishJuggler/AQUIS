"""Assistant page — station-locked natural-language Q&A over live AQUIS data."""

import pandas as pd
import streamlit as st

from _assistant import (
    StationAssistant,
    get_df,
    ollama_model,
    ollama_status,
)
from _utils import selected_from_sidebar

st.set_page_config(page_title="AQUIS — assistant", page_icon=":material/smart_toy:", layout="wide")
st.title("Assistant — groundwater data in plain language")

df = get_df()
status = ollama_status()
model_name = status["model"]
up = status["server"] and model_name in status["models"]

if not status["server"]:
    st.warning(
        "Ollama server not reachable at the configured host. Start it with "
        "`ollama serve` in another terminal; the assistant needs it for chat answers."
    )
    if status.get("error"):
        st.code(f"host={status['error']}", language="text")
elif not up:
    st.info(f"Ollama reachable, but model **`{model_name}`** is not pulled. Run:")
    st.code(f"ollama pull {model_name}", language="bash")

# --- District & station selection (global sidebar in app.py — always visible) ---
recency = df.groupby("Station")["time"].max().sort_values(ascending=False)
stations = [str(s) for s in recency.index]
selected_district, station = selected_from_sidebar(df, stations)

# --- Chat ---
assistant = StationAssistant()

if st.session_state.get("as_pinned_station") != station:
    st.session_state.as_pinned_station = station
    st.session_state.as_messages = [
        {"role": "assistant",
         "content": f"Hello! I can answer about **{station}** ({selected_district}).\n\n"
                    "Try: *latest level*, *trend over 30 days*, *what's driving the level?*, "
                    "*what was the level on 15 March 2023?*, or *suggest precautions*."}
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
            answer = (
                f"Ollama not available. Start it with `ollama serve` and ensure "
                f"`{model_name}` is pulled — then ask again."
            )
        else:
            with st.spinner("Querying AQUIS data + generating answer (local model, ~15-60s)..."):
                try:
                    res = assistant.answer(prompt, station, history=history)
                    answer = res.get("answer", "Sorry, could not generate an answer.")
                except Exception as e:  # noqa: BLE001
                    answer = f"Assistant error: `{e}`"
            st.markdown(answer)
    st.session_state.as_messages.append({"role": "assistant", "content": answer})
