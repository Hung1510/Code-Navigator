// main.js — frontend logic. Talks to the Rust backend via the global Tauri API
// (enabled by "withGlobalTauri": true in tauri.conf.json), which runs the
// codenavigator CLI and returns its stdout/stderr.

const { invoke } = window.__TAURI__.core;

const $ = (id) => document.getElementById(id);
const repoEl = $("repo");
const queryEl = $("query");
const modeEl = $("mode");
const rerankEl = $("rerank");
const topkEl = $("topk");
const statusEl = $("status");
const resultsEl = $("results");

let action = "search"; // or "ask"

// --- action toggle (Search / Ask) ---
document.querySelectorAll("#action-seg button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("#action-seg button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    action = b.dataset.action;
    $("btn-run").textContent = action === "ask" ? "Ask" : "Run";
  });
});

function setStatus(msg, kind = "info") {
  if (!msg) { statusEl.classList.add("hidden"); return; }
  statusEl.textContent = msg;
  statusEl.className = `status ${kind}`;
}

function busy(on) {
  $("btn-run").disabled = on;
  $("btn-index").disabled = on;
  $("btn-run").classList.toggle("loading", on);
}

async function callCli(args) {
  const res = await invoke("run_codenavigator", { args });
  return res; // { ok, stdout, stderr }
}

// --- Index ---
$("btn-index").addEventListener("click", async () => {
  const repo = repoEl.value.trim();
  if (!repo) return setStatus("Enter a repo path first.", "error");
  busy(true);
  setStatus("Indexing… first run downloads the embedding model (~130MB).", "info");
  try {
    const res = await callCli(["index", repo]);
    // index prints progress to stdout; show the last meaningful line.
    const line = (res.stdout || res.stderr || "").trim().split("\n").filter(Boolean).pop();
    setStatus(res.ok ? (line || "Index built.") : (res.stderr || "Index failed."), res.ok ? "ok" : "error");
  } catch (e) {
    setStatus(String(e), "error");
  } finally {
    busy(false);
  }
});

// --- Run (search or ask) ---
async function run() {
  const repo = repoEl.value.trim();
  const q = queryEl.value.trim();
  if (!repo) return setStatus("Enter a repo path first.", "error");
  if (!q) return setStatus("Enter a question.", "error");

  const args = [action, repo, q, "--json", "--mode", modeEl.value, "-k", String(topkEl.value || 8)];
  if (!rerankEl.checked) args.push("--no-rerank");

  busy(true);
  setStatus(action === "ask" ? "Retrieving + asking Claude…" : "Retrieving…", "info");
  try {
    const res = await callCli(args);
    if (!res.ok) { setStatus(res.stderr || "Command failed.", "error"); return; }
    const data = JSON.parse(res.stdout);
    if (action === "ask") renderAnswer(data);
    else renderHits(data);
    setStatus("", "info");
  } catch (e) {
    setStatus("Could not parse results: " + String(e), "error");
  } finally {
    busy(false);
  }
}
$("btn-run").addEventListener("click", run);
queryEl.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
repoEl.addEventListener("keydown", (e) => { if (e.key === "Enter") queryEl.focus(); });

// --- rendering ---
function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function hitCard(h) {
  return `
    <article class="card">
      <div class="card-head">
        <span class="locator">${esc(h.locator)}</span>
        <span class="score">${h.score.toFixed(4)}</span>
      </div>
      <pre class="code"><code>${esc(h.text)}</code></pre>
    </article>`;
}

function renderHits(hits) {
  if (!hits.length) { resultsEl.innerHTML = `<div class="empty"><p>No matches.</p></div>`; return; }
  resultsEl.innerHTML = hits.map(hitCard).join("");
}

function renderAnswer(data) {
  const hits = data.hits || [];
  resultsEl.innerHTML = `
    <article class="answer">
      <h2>Answer</h2>
      <div class="answer-body">${esc(data.answer || "").replace(/\n/g, "<br>")}</div>
    </article>
    <h3 class="sources-h">Sources (${hits.length})</h3>
    ${hits.map(hitCard).join("")}`;
}
