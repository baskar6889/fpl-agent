"""Small session-only Streamlit chat UI for the FPL ranking agent."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RAW_DIR = ROOT / "data" / "raw"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fpl_agent.agents import RankingAgent
from fpl_agent.application import build_backend, refresh_backend


def _build_agent_from_latest() -> tuple[RankingAgent, Path]:
    backend = build_backend(require_agent=True)
    assert backend.agent is not None
    return backend.agent, backend.dataset_path


def _clear_agent_state() -> None:
    st.session_state.agent = None
    st.session_state.dataset_path = None


def _load_agent_if_available() -> None:
    if "agent" in st.session_state:
        return
    try:
        agent, data_path = _build_agent_from_latest()
    except FileNotFoundError:
        st.session_state.agent = None
        st.session_state.dataset_path = None
    except Exception as exc:
        _clear_agent_state()
        st.session_state.startup_error = str(exc)
    else:
        st.session_state.agent = agent
        st.session_state.dataset_path = str(data_path)
        st.session_state.startup_error = None


st.set_page_config(page_title="FPL Agent", page_icon="⚽")
st.title("FPL Agent")
st.caption("Ask a question about ranking FPL players.")

if "messages" not in st.session_state:
    st.session_state.messages = []
_load_agent_if_available()

with st.sidebar:
    st.subheader("Data")
    if st.session_state.dataset_path:
        st.caption(f"Loaded dataset: {Path(st.session_state.dataset_path).name}")
    else:
        st.info("FPL data has not been loaded yet. Refresh data to get started.")
    if st.session_state.get("startup_error"):
        st.error(f"Agent initialization failed: {st.session_state.startup_error}")

    if st.button("Refresh Data", type="primary", use_container_width=True):
        _clear_agent_state()
        st.session_state.startup_error = None
        try:
            with st.status("Refreshing FPL data…", expanded=False) as status:
                backend = refresh_backend(raw_dir=RAW_DIR, require_agent=True)
                assert backend.agent is not None
                st.session_state.agent = backend.agent
                st.session_state.dataset_path = str(backend.dataset_path)
                st.session_state.startup_error = None
                status.update(label="FPL data refreshed.", state="complete")
        except Exception as exc:
            _clear_agent_state()
            st.session_state.startup_error = str(exc)
            st.error(f"Data refresh failed. The agent is unavailable: {exc}")
        st.rerun()

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about FPL players"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = st.session_state.agent
    if agent is None:
        response = "FPL data or agent configuration is unavailable. Check the message in the sidebar and refresh data after resolving it."
    else:
        try:
            with st.spinner("Thinking…"):
                response = agent.respond(prompt).response
        except Exception as exc:
            response = f"The request failed: {exc}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
