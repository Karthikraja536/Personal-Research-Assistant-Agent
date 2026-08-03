# Personal Research Assistant Agent

An AI agent that researches a question by deciding, on its own, whether to search the live web or academic papers on arXiv — then reads the results and repeats until it has enough to give a grounded, cited answer.

Built with **LangGraph**: the agent is an explicit graph of nodes and edges, not a hidden black box, so the reasoning loop is fully inspectable.

## Overview

Most "AI research tools" are retrieval pipelines: fetch once, generate an answer, done. This project is an **agent** in the stricter sense — the language model itself decides at each step whether it has enough information or needs to call a tool, which tool to call, and when to stop. That decision loop, not the retrieval, is what makes it agentic.

## Features

- **Autonomous tool selection** — the model chooses between web search and arXiv search based on the question, with no hard-coded routing logic.
- **Multi-step reasoning** — the agent can call tools more than once, refining its search before answering.
- **Transparent reasoning trace** — every tool call, its inputs, and its output are shown in the UI, not hidden behind the final answer.
- **Cited answers** — the system prompt requires the agent to reference where each claim came from.
- **No exposed credentials** — the API key is read from environment configuration only; it is never entered or displayed in the UI.
- **Fails gracefully** — missing configuration, tool errors, or API errors all return a readable message instead of crashing the app.

## Demo

*Add a screenshot or short screen recording of the app here (e.g. `docs/demo.png`) before sharing this repository publicly.*

## Architecture

```
User question
      |
      v
Agent node (LLM) ----------------------> Final answer (with sources)
      |          ^
      v          |
  Tools node     |
 (web / arXiv)   |
      |          |
      v          |
  Result --------
 (fed back to the agent node, loop continues)
```

The loop is built as a two-node LangGraph graph:

| Step | Graph element |
|---|---|
| Ask the LLM what to do next | `agent` node (`call_model`) |
| Run whichever tool the LLM requested | `tools` node (`ToolNode`) |
| Decide: call another tool, or stop | Conditional edge (`should_continue`) |
| Loop back after a tool runs | Edge from `tools` → `agent` |

## Tech Stack

| Component | Choice | Notes |
|---|---|---|
| Agent orchestration | LangGraph (`StateGraph`) | Explicit graph rather than a hidden agent-executor abstraction |
| LLM + tool calling | Groq — `openai/gpt-oss-120b`, via `langchain-groq` | Fast inference, free tier, reliable tool-calling |
| Web search | `ddgs` (DuckDuckGo) | No API key required |
| Academic search | `arxiv` | Official arXiv API wrapper, no key required |
| UI | Streamlit | Single-page app, no separate frontend build |

## Getting Started

Requires Python 3.10+. Works the same way on Windows, macOS, and Linux.

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure your API key**

Copy the example environment file:
```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
copy .env.example .env
```
Get a free key from [console.groq.com/keys](https://console.groq.com/keys) and add it to `.env`:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

**3. Run the app**
```bash
streamlit run app.py
```
Opens automatically at `http://localhost:8501`.

## Usage

Type a research question, or click one of the example prompts in the sidebar. Click **Research** and the agent will search, reason, and return a cited answer. Expand **"See the agent's reasoning"** below the answer to inspect exactly which tools were called, with what inputs, and what each one returned.

## Design Notes

A few deliberate choices worth calling out:

- **Core `StateGraph`, not the `create_agent` shortcut.** LangGraph reached a stable v1.0 in October 2025; the older `create_react_agent` helper is now deprecated in favor of LangChain's `create_agent`. Either approach works, but building the graph explicitly keeps every node and edge visible rather than hidden behind one function call. The one-line equivalent is included as a comment at the bottom of `agent.py`.
- **Tools defined with `@tool`.** The decorator generates the function-calling schema automatically from each function's type hints and docstring — no hand-written JSON schema to maintain.
- **The LLM client is built lazily**, on first use rather than at import time. Constructing it eagerly would mean the app crashes on startup if the API key isn't configured yet, before any error message can be shown.
- **Markdown output is sanitized defensively.** The system prompt instructs the model to avoid raw HTML, and the UI additionally strips any stray `<br>` tags before rendering, so malformed model output never surfaces as visible artifacts.

## Project Structure

```
research-assistant-agent/
├── app.py              # Streamlit UI — no agent logic here
├── agent.py             # The agent: graph, nodes, tools, system prompt
├── tools.py              # Search implementations (web + arXiv)
├── requirements.txt
├── .env.example
└── README.md
```

`agent.py` and `tools.py` have no Streamlit dependency, so the agent can be run and tested independently of the UI.

## Testing

Run the agent directly from the command line, without the UI:
```bash
python agent.py
```
This executes a sample query and prints the full tool-call trace and final answer to the terminal.

## Roadmap

- [ ] Additional tools (Wikipedia lookup, calculator) — pattern is `@tool`-decorate a function and add it to `TOOLS`
- [ ] PDF upload support, so the agent can research user-supplied documents alongside the live web
- [ ] Conversation memory via a LangGraph checkpointer, to support follow-up questions
- [ ] Structured, machine-readable citations (`{answer, sources: [...]}`) instead of free-text references


## License

MIT — see `LICENSE`. (Add a `LICENSE` file with the standard MIT text if one isn't present yet.)

## Author

**Karthikraja V**
[GitHub](https://github.com/Karthikraja536) · [LinkedIn](https://linkedin.com/in/karthikraja06)
