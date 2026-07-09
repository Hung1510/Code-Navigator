# CodeNavigator desktop

A Tauri 2 desktop shell over the `codenav` engine. The Rust backend runs the
Python `codenav` CLI with `--json`; a lightweight static web frontend renders
the results. Search shows retrieved code; Ask sends it to Claude for a grounded,
cited answer.

```
  ui/ (HTML/JS/CSS)  ──invoke──►  src-tauri (Rust)  ──subprocess──►  python -m codenavigator --json
        renders            run_codenavigator                     retrieval / rerank / LLM
```

## Prerequisites

1. **CodeNavigator installed and importable** by the Python on your PATH:
   ```bash
   cd ..            # the CodeNavigator project root
   pip install -e ".[treesitter]"
   python -m codenavigator --help    # must work
   ```
   If CodeNavigator lives in a venv, point the app at that interpreter:
   `export CODENAVIGATOR_PYTHON=/path/to/venv/bin/python`
2. **Rust + Tauri prerequisites** — see https://v2.tauri.app/start/prerequisites/
   (Rust toolchain, and on Linux the webkit2gtk / libsoup system packages).
3. **Node.js** (for the Tauri CLI).

## Setup

```bash
cd desktop
npm install
npm run icons          # one-time: generates src-tauri/icons/* from ui/icon.png
npm run dev            # launch the app (first Rust build takes a few minutes)
```

For `ask` to return answers, set your key before launching:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
npm run dev
```

Build a distributable bundle with `npm run build`.

## How it fits together

- **`src-tauri/src/main.rs`** — exposes one command, `run_codenavigator(args)`, which
  runs `python -m codenavigator <args>` on a blocking thread (so the window stays
  responsive during indexing or an LLM call) and returns stdout/stderr.
- **`ui/main.js`** — builds the CLI args from the form (repo, query, mode,
  rerank, k), calls the command through the global Tauri API, parses the JSON,
  and renders hit cards or the answer + sources.
- **`ui/index.html` / `ui/styles.css`** — the interface. No framework, no
  bundler: `withGlobalTauri` exposes `window.__TAURI__`, so the frontend is
  plain static files served straight from `frontendDist`.

## Notes & next steps

- **Streaming:** the backend captures all CLI output at once, so a long index
  build shows only the final status line, not live progress. To stream, switch
  `run_codenavigator` to spawn the child and emit stdout lines as Tauri events.
- **Real sidecar for distribution:** dev mode calls whatever `python` is on
  PATH. To ship a self-contained app, bundle CodeNavigator with pyinstaller and wire
  it in via Tauri's `externalBin` sidecar mechanism instead of a bare
  subprocess.
- **Native folder picker:** the repo is a text field today; add
  `@tauri-apps/plugin-dialog` for a native directory chooser.
