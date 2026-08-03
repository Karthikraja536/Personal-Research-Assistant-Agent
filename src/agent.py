"""
agent.py
--------
The "agent" part of the project: an LLM that decides, on its own, when
to call a tool and when to answer -- built with LangGraph, so the loop
is an explicit graph (nodes + edges) instead of hidden inside a
framework you can't see into.

LangGraph models this as:
  - an "agent" node   -> ask the LLM what to do next
  - a  "tools" node   -> run whatever tool(s) the LLM asked for
  - an edge that loops from "tools" back to "agent", until the LLM
    stops calling tools and just writes an answer

This uses LangGraph's core StateGraph directly (not the newer, higher
-level langchain.agents.create_agent one-liner), so the graph mechanics
-- nodes, edges, state -- stay visible instead of hidden behind a
single function call.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# The actual search logic lives in tools.py and doesn't change -- only
# how it's *described to the model* changes (see the @tool wrappers
# below).
from tools import web_search as _web_search, arxiv_search as _arxiv_search

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a Personal Research Assistant Agent.

You have two tools:
- web_search: current events, general knowledge, anything time-sensitive
- arxiv_search: academic papers, technical/scientific topics

Guidelines:
1. Think about what the question needs, then pick ONE tool to start.
2. Only call a tool again if the first result was genuinely unhelpful.
3. As soon as you have enough information to answer, STOP calling tools
   and write the answer. 2-3 good results are usually enough.
4. Reference where each claim came from (source name or link).
5. Never invent facts, sources, or paper titles you did not actually
   retrieve from a tool call.
6. Write your final answer in plain Markdown only. Never use raw HTML
   tags anywhere, including <br> inside table cells -- if a table cell
   would need multiple lines or bullet points, keep that cell to one
   short line instead, or use a bullet list section rather than a
   table for that content.
"""


# --- Tools, LangChain-style ------------------------------------------
# The @tool decorator builds the function-calling schema automatically
# from each function's type hints and docstring -- no hand-written
# JSON schema required.

@tool
def web_search(query: str) -> str:
    """Search the live web for current events, news, general knowledge,
    or anything time-sensitive. Use this when the question is not
    primarily academic."""
    return _web_search(query)


@tool
def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search arXiv.org for academic and scientific papers. Use this for
    research topics or technical/scientific questions."""
    return _arxiv_search(query, max_results)


TOOLS = [web_search, arxiv_search]


# --- LLM: built lazily, not at import time --------------------------
# ChatGroq validates GROQ_API_KEY as soon as it's constructed. Building
# it here at module level would mean simply *importing* this file
# crashes if the key isn't set yet -- before app.py ever gets a chance
# to show a friendly error. Building it on first use avoids that.

_llm_with_tools = None


def get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = ChatGroq(model=MODEL, temperature=0.3)
        _llm_with_tools = llm.bind_tools(TOOLS)
    return _llm_with_tools


# --- Graph nodes --------------------------------------------------------

def call_model(state: MessagesState):
    """The 'agent' node: ask the LLM what to do next, given the
    conversation (including any tool results) so far."""
    response = get_llm_with_tools().invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState):
    """The conditional edge: decides whether to loop back for a tool
    call, or stop because the model wrote a final answer."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# --- Build the graph ------------------------------------------------------
# Building and compiling the graph itself needs no API key -- call_model
# only touches get_llm_with_tools() once the graph actually runs.

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", ToolNode(TOOLS))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")  # after tools, go back to agent -> the loop

graph = graph_builder.compile()


def run_agent(user_query: str):
    """
    Run the agent for a single research question.

    Returns:
        (final_answer: str, trace: list[dict]) -- trace is the
        step-by-step record of tool calls, shown in the UI as the
        agent's "reasoning". Never raises -- any failure comes back as
        a readable string instead.
    """
    if not os.environ.get("GROQ_API_KEY"):
        return (
            "GROQ_API_KEY is not set. Add it to a .env file in the "
            "project folder (see .env.example) and restart the app.",
            [],
        )

    try:
        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                {"role": "user", "content": user_query},
            ]
        }
        result = graph.invoke(initial_state)

        # Rebuild a step-by-step trace for the UI by matching each tool
        # result back to the tool call that triggered it.
        trace = []
        pending_calls = {}
        step = 0
        for msg in result["messages"]:
            cls_name = msg.__class__.__name__
            if cls_name == "AIMessage" and getattr(msg, "tool_calls", None):
                step += 1
                for call in msg.tool_calls:
                    pending_calls[call["id"]] = {
                        "step": step,
                        "tool": call["name"],
                        "input": call["args"],
                    }
            elif cls_name == "ToolMessage":
                info = pending_calls.get(msg.tool_call_id, {})
                trace.append({
                    "step": info.get("step", step),
                    "tool": info.get("tool", getattr(msg, "name", "unknown")),
                    "input": info.get("input", {}),
                    "output": msg.content,
                })

        final_answer = result["messages"][-1].content
        return final_answer, trace

    except Exception as e:
        return f"Something went wrong and I couldn't complete this: {e}", []


if __name__ == "__main__":
    # Quick manual test -- run `python agent.py` to sanity-check the
    # graph without touching the Streamlit UI.
    test_query = "What is TON 618 and how big is it?"
    answer, trace = run_agent(test_query)

    print("=== TRACE ===")
    for t in trace:
        print(f"[step {t['step']}] {t['tool']}({t['input']})")
        print(f"  -> {t['output'][:200]}...\n")

    print("=== ANSWER ===")
    print(answer)
