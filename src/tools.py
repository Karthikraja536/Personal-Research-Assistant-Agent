"""
tools.py
--------
Every tool the agent can call, plus the JSON "schemas" that describe them
to the LLM (this is how function calling works: you describe the tool in
plain language + a parameter spec, and the model decides when to use it).

Each tool function:
  - takes plain arguments
  - does the actual work (search, fetch, etc.)
  - returns a STRING (tool results are always sent back to the LLM as text)
  - never raises an unhandled exception (the agent loop would crash) —
    catch errors and return them as a string instead, so the agent can
    see the failure and try something else.
"""

from ddgs import DDGS
import arxiv


def web_search(query: str, max_results: int = 8) -> str:
    """Search the live web via DuckDuckGo. No API key required."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No web results found for this query."

        lines = []
        for r in results:
            title = r.get("title", "Untitled")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"- {title}: {body}\n  Source: {href}")
        return "\n".join(lines)
    except Exception as e:
        return f"web_search failed: {e}"


def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv.org for academic papers. No API key required."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        lines = []
        for paper in client.results(search):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            summary = paper.summary.replace("\n", " ")[:300]
            lines.append(
                f"- {paper.title} ({paper.published.year}) — {authors}\n"
                f"  Summary: {summary}...\n"
                f"  Link: {paper.entry_id}"
            )

        return "\n".join(lines) if lines else "No papers found for this query."
    except Exception as e:
        return f"arxiv_search failed: {e}"


# Maps tool name -> actual Python function, so the agent loop can call
# whatever the model asks for by name.
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "arxiv_search": arxiv_search,
}

# OpenAI/Groq-compatible function-calling schemas. The "description" fields
# matter a lot — this is how the model decides WHICH tool to use. Vague
# descriptions -> vague tool choices. Be specific about when each tool
# applies.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web for current events, news, general "
                "knowledge, product/company info, or anything time-sensitive. "
                "Use this when the question is not primarily academic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": (
                "Search arXiv.org for academic and scientific papers. Use "
                "this for research topics, technical/scientific questions, "
                "or when the user explicitly asks about papers or studies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for arXiv.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of papers to return. Default 3.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]
