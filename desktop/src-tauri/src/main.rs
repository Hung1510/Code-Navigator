// main.rs — the Tauri backend.
//
// The heavy lifting (chunking, embedding, retrieval, reranking, the LLM call)
// all lives in the Python `codenavigator` package. This backend is a thin bridge: it
// exposes one command, `run_codenavigator`, that shells out to `python -m codenavigator ...`
// and hands the captured stdout/stderr back to the web frontend, which parses
// the `--json` output and renders it.
//
// This is the "Python sidecar" pattern. For a distributable build you'd bundle
// codenavigator as a real Tauri sidecar (pyinstaller -> externalBin); for development
// this calls whatever `python` is on PATH (override with CODENAVIGATOR_PYTHON).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::Command;

use serde::Serialize;

#[derive(Serialize)]
struct CliResult {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn python_bin() -> String {
    std::env::var("CODENAVIGATOR_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

/// Run `python -m codenavigator <args>` off the async runtime so the UI stays
/// responsive during a long index build or an LLM round-trip.
#[tauri::command]
async fn run_codenavigator(args: Vec<String>) -> CliResult {
    tauri::async_runtime::spawn_blocking(move || {
        match Command::new(python_bin()).arg("-m").arg("codenavigator").args(&args).output() {
            Ok(o) => CliResult {
                ok: o.status.success(),
                stdout: String::from_utf8_lossy(&o.stdout).to_string(),
                stderr: String::from_utf8_lossy(&o.stderr).to_string(),
            },
            Err(e) => CliResult {
                ok: false,
                stdout: String::new(),
                stderr: format!(
                    "Failed to launch '{}': {e}. Is codenavigator installed and on PATH? \
                     Set CODENAVIGATOR_PYTHON to point at the right interpreter.",
                    python_bin()
                ),
            },
        }
    })
    .await
    .unwrap_or_else(|e| CliResult {
        ok: false,
        stdout: String::new(),
        stderr: format!("join error: {e}"),
    })
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![run_codenavigator])
        .run(tauri::generate_context!())
        .expect("error while running codenavigator desktop");
}
