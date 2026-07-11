# demo.tape — the README hero asset, as code.
#
# Renders docs/demo.gif with VHS (github.com/charmbracelet/vhs). Because the
# GIF is generated from this file rather than screen-recorded, it can never
# quietly drift out of sync with what the tool actually prints — re-run the
# tape and the README is true again.
#
#   docker run --rm -v ${PWD}:/vhs ghcr.io/charmbracelet/vhs demo/demo.tape
#
# or, with vhs installed locally (needs ttyd + ffmpeg on PATH):
#
#   vhs demo/demo.tape
#
# ---------------------------------------------------------------------------
# The story this tells, in order, and why:
#
#   1. It indexes ITSELF. No fixture repo, no setup, nothing to take on faith —
#      the thing in the GIF is the thing you just cloned.
#   2. Search with file:line citations. Table stakes; every RAG tool does this.
#      Shown fast, because it is not the reason to care.
#   3. `impact` — transitive blast radius. THIS is the frame that earns the star.
#      A viewer who has ever broken prod by changing a signature understands it
#      instantly, and knows grep cannot do it.
#   4. `tests` — with the call chain. The payoff line: it does not just say
#      WHICH tests, it says WHY, through the helper that connects them.
#
# The whole point is to get to frame 3 fast. Everything before it is setup.
# ---------------------------------------------------------------------------

Output docs/demo.gif

Require codenav

Set Shell bash
Set FontSize 17
Set Width 1300
Set Height 760
Set Padding 22
Set TypingSpeed 45ms
Set LoopOffset 20%

# --- Setup, hidden: warm the model cache and clear any prior index so the
# --- recorded run is honest about timing but not about the one-time download.
Hide
Type "clear && rm -rf .codenavigator"
Enter
Sleep 500ms
Show

# --- 1. Index itself ------------------------------------------------------
Type "# CodeNavigator, indexing its own source. Fully local — no API key."
Sleep 1500ms
Enter
Type "codenav index ."
Sleep 600ms
Enter
Sleep 6s

Type "clear"
Enter
Sleep 300ms

# --- 2. Search: cited retrieval (brief — everyone does this) ---------------
Type "# Ask it something. Answers cite file:line."
Sleep 1200ms
Enter
Type "codenav search . 'how are vector and keyword results fused?' -k 3"
Sleep 800ms
Enter
Sleep 6s

Type "clear"
Enter
Sleep 300ms

# --- 3. Impact: the hero frame --------------------------------------------
Type "# Now the part grep can't do. What breaks if I change this function?"
Sleep 2s
Enter
Type "codenav impact . _base_ranking --depth 2"
Sleep 800ms
Enter
Sleep 8s

Type "clear"
Enter
Sleep 300ms

# --- 4. Tests: which tests, and WHY ---------------------------------------
Type "# And which tests actually reach it — with the chain that gets them there."
Sleep 2s
Enter
Type "codenav tests . _base_ranking"
Sleep 800ms
Enter
Sleep 8s

# --- Land the plane -------------------------------------------------------
Type "# Same engine, over MCP: your assistant gets this too."
Sleep 2500ms
Enter
Sleep 2s
