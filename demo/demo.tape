# demo.tape — the README hero asset, as code.
#
# Renders docs/demo.gif. See demo/README.md for how to run it.
#
# Timing note: the graph commands are instant, so Sleep is fine for those —
# it's a deliberate "let the viewer read it" pause, not a guess. But `index`
# takes an unpredictable amount of time (it depends on your CPU), so that one
# uses `Wait`, which blocks until the output actually appears. A `Sleep` there
# would either cut the output off on a slow machine or dead-air on a fast one.
#
# ---------------------------------------------------------------------------
# The story, in order, and why:
#   1. It indexes ITSELF — nothing to take on faith, no fixture repo.
#   2. Search with file:line citations. Shown fast: every RAG tool does this.
#   3. `impact` — transitive blast radius. The frame that earns the star.
#   4. `tests` — with the call chain. Not just WHICH tests, but WHY.
# Get to frame 3 quickly. Everything before it is setup.
# ---------------------------------------------------------------------------

Output docs/demo.gif

Require codenav

Set Shell bash
Set FontSize 17
Set Width 1300
Set Height 760
Set Padding 20
Set TypingSpeed 45ms
Set Framerate 24
Set LoopOffset 20%

# --- Hidden setup ----------------------------------------------------------
# A short prompt (the default is long and noisy), and a clean slate so the
# index we record is a real cold build — of the INDEX, not of the model cache.
# The model must already be downloaded before you record. See demo/README.md.
Hide
Type `export PS1="$ "`
Enter
Type "clear && rm -rf .codenavigator"
Enter
Sleep 500ms
Show

# --- 1. Index itself -------------------------------------------------------
Type "# CodeNavigator, indexing its own source. Local — no API key, no cloud."
Sleep 1500ms
Enter
Type "codenav index ."
Sleep 600ms
Enter

# Block until the last line of a successful index actually appears.
Wait+Screen@90s /call sites/
Sleep 2s

Type "clear"
Enter
Sleep 400ms

# --- 2. Search: cited retrieval (brief) ------------------------------------
Type "# Ask in plain language. Every answer cites file:line."
Sleep 1400ms
Enter
Type "codenav search . 'how are vector and keyword results fused?' -k 3"
Sleep 700ms
Enter
Sleep 3s
Sleep 3500ms

Type "clear"
Enter
Sleep 400ms

# --- 3. Impact: the hero frame ---------------------------------------------
Type "# Now the part grep can't do: what breaks if I change this function?"
Sleep 2200ms
Enter
Type "codenav impact . _base_ranking --depth 2"
Sleep 700ms
Enter
Sleep 6s

Type "clear"
Enter
Sleep 400ms

# --- 4. Tests: which tests, and WHY ----------------------------------------
Type "# And which tests reach it — with the chain that gets them there."
Sleep 2200ms
Enter
Type "codenav tests . _base_ranking"
Sleep 700ms
Enter
Sleep 6500ms

# --- Land the plane --------------------------------------------------------
Type "# Same engine over MCP. Your AI assistant gets all of this too."
Sleep 2500ms
Enter
Sleep 1500ms
