"""
app.py
------
Streamlit front-end. Deliberately thin -- all the actual "agent" logic
lives in agent.py so it can be tested and reasoned about on its own.

The Groq API key is read only from the environment (.env file) -- it is
never shown or entered in the UI. Run with:  streamlit run app.py
"""

import json
import os

import streamlit as st
from dotenv import load_dotenv

from agent import run_agent, MODEL

load_dotenv()  # picks up GROQ_API_KEY from a local .env file

st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="🔎",
    layout="centered",
)

st.title("🔎 Personal Research Assistant Agent")
st.caption(
    "An agent that reasons about your question, then searches the web "
    "and/or arXiv to answer it."
)

EXAMPLE_QUERIES = [
    "Latest advances in LLM agent memory",
    "How does retrieval-augmented generation work?",
    "Recent breakthroughs in quantum computing",
]

with st.sidebar:
    st.subheader("Try an example")
    for example in EXAMPLE_QUERIES:
        if st.button(example, use_container_width=True):
            st.session_state.query_input = example

    st.divider()
    st.caption(f"Agent: LangGraph  ·  Model: `{MODEL}`")
    st.caption("Tools: Web search (DuckDuckGo) · arXiv")

query = st.text_input(
    "What do you want to research?",
    key="query_input",
    placeholder="e.g. What are the latest approaches to LLM agent memory?",
)

run_clicked = st.button("Research", type="primary")

if run_clicked and query:
    if not os.environ.get("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY is not set. Add it to a `.env` file in the "
            "project folder (see `.env.example`), then restart the app."
        )
    else:
        with st.spinner("Researching..."):
            try:
                answer, trace = run_agent(query)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                answer, trace = None, []

        if answer:
            st.subheader("Answer")
            st.markdown(answer)

            if trace:
                with st.expander(f"🧠 See the agent's reasoning ({len(trace)} tool call(s))"):
                    for t in trace:
                        st.markdown(f"**Step {t['step']} — called `{t['tool']}`**")
                        st.code(json.dumps(t["input"]), language="json")
                        st.text(t["output"][:600])
                        st.divider()
elif run_clicked and not query:
    st.warning("Type a research question first.")
