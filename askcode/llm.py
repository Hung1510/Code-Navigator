"""
llm.py — the "generation" half of retrieval-augmented generation.

Retrieval found the relevant code. Now we hand those chunks to Claude and ask
it to answer *grounded in them*, citing file:line locators so you can verify
the answer instead of trusting it. The system prompt is deliberately strict:
answer from the provided context, and say so when the context doesn't cover it.
That discipline is what separates a useful code assistant from a confident
hallucinator.

Uses the standard Anthropic Messages API. Set ANTHROPIC_API_KEY in your env.
"""

from __future__ import annotations

import os

from .store import SearchHit

DEFAULT_MODEL = os.environ.get("ASKCODE_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a code assistant answering questions about a specific codebase.
You are given retrieved code chunks, each labelled with its file:line locator.

Rules:
- Answer ONLY from the provided chunks. Do not invent files, functions, or behavior.
- Some chunks are marked "\u2190 called by X": they were pulled in via the call
  graph because X (one of the top matches) calls them. Use them to explain what
  the matched code actually does, and cite them like any other chunk.
- Cite the locator (path:start-end) for every claim about the code.
- If the chunks don't contain the answer, say so plainly and suggest what to search for next.
- Be concise and concrete. Show the relevant lines when it helps.
"""


def _build_context(hits: list[SearchHit]) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"--- {h.locator()}  [score={h.score:.3f}] ---\n{h.text}"
        )
    return "\n\n".join(blocks)


def answer(question: str, hits: list[SearchHit], model: str = DEFAULT_MODEL) -> str:
    """Call Claude with the retrieved context. Returns the answer text."""
    if not hits:
        return "No relevant code found in the index. Try re-indexing or rephrasing."

    try:
        import anthropic
    except ImportError:
        return (
            "The `anthropic` package isn't installed.\n"
            "Retrieved chunks:\n\n" + _build_context(hits)
        )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return (
            "ANTHROPIC_API_KEY not set — showing raw retrieval instead of a synthesized answer.\n\n"
            + _build_context(hits)
        )

    client = anthropic.Anthropic()
    user_msg = (
        f"Question: {question}\n\n"
        f"Retrieved code chunks:\n\n{_build_context(hits)}\n\n"
        f"Answer the question using only these chunks, citing locators."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
