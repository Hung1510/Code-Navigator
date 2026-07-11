"""
End-to-end MCP test: spawns the real server over stdio and drives it with the
official MCP client. Proves tool discovery + tool calls work over the actual
JSON-RPC protocol, not just that the Python functions are callable.

Uses a fake embedder (injected via CODENAVIGATOR_FAKE_EMBED=1) so no model
download is needed. Run:  python tests/test_mcp.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_SRC = {
    "svc/tokens.py": (
        "import time\n\n"
        "def issue_jwt(user):\n"
        "    payload = {'sub': user.id, 'exp': time.time() + 3600}\n"
        "    return sign(payload)\n\n"
        "def sign(payload):\n"
        "    return hmac(SECRET, payload)\n"
    ),
    "svc/auth.py": (
        "from .tokens import issue_jwt\n\n"
        "class AuthService:\n"
        "    def login(self, user):\n"
        "        token = issue_jwt(user)\n"
        "        self.sessions.add(token)\n"
        "        return token\n"
    ),
    "svc/api.py": (
        "from .auth import AuthService\n\n"
        "def handle_login(req):\n"
        "    return AuthService().login(req.user)\n"
    ),
}


def make_repo(root: Path):
    for rel, src in REPO_SRC.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)


async def run(repo: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env.update({
        "CODENAVIGATOR_REPO": str(repo),
        "CODENAVIGATOR_FAKE_EMBED": "1",   # skip the model download
        "CODENAVIGATOR_RERANK": "0",       # skip the cross-encoder download
        "PYTHONPATH": str(project_root),
    })
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "codenavigator.mcp_server"], env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            print("tools discovered:", sorted(names))
            assert {"search_code", "ask_codebase", "get_definition",
                    "find_callers", "find_callees"} <= names

            # every tool must carry a description (the model reads these)
            for t in tools.tools:
                assert t.description and len(t.description) > 20, t.name

            r = await session.call_tool("search_code", {"query": "login", "k": 3})
            text = r.content[0].text
            print("\n[search_code]\n" + text[:220])
            assert "auth.py" in text and ":" in text

            r = await session.call_tool("get_definition", {"symbol": "issue_jwt"})
            text = r.content[0].text
            print("\n[get_definition]\n" + text)
            assert "tokens.py" in text

            r = await session.call_tool("find_callers", {"symbol": "issue_jwt"})
            text = r.content[0].text
            print("\n[find_callers]\n" + text)
            assert "AuthService.login" in text

            r = await session.call_tool("find_callees", {"symbol": "AuthService.login"})
            text = r.content[0].text
            print("\n[find_callees]\n" + text)
            assert "issue_jwt" in text

            # graph expansion: asking about login must pull in issue_jwt from
            # the OTHER file, annotated as a call-graph edge.
            r = await session.call_tool("ask_codebase",
                                        {"question": "how does AuthService login work", "k": 3})
            text = r.content[0].text
            print("\n[ask_codebase]\n" + text[:400])
            assert "tokens.py" in text and "called by" in text

            # budget: response must respect the cap
            assert len(text) <= 6000 + 500, f"budget exceeded: {len(text)}"

    print("\nMCP end-to-end: PASSED")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "repo"
        repo.mkdir()
        make_repo(repo)
        asyncio.run(run(repo))
