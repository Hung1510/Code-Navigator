"""
cli.py — the command-line interface.

    codenav index  <repo> [--full]
    codenav ask    <repo> "question" [--mode] [--no-rerank] [--json]
    codenav search <repo> "question" [--mode] [--no-rerank] [--json]

`ask` and `search` retrieve with the two-stage pipeline (hybrid + optional
cross-encoder rerank). `--json` emits structured output for scripting or the
desktop app; otherwise output is human-readable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _embedder(model: str | None):
    from .embedder import DEFAULT_MODEL, FastEmbedder
    return FastEmbedder(model_name=model or DEFAULT_MODEL)


def _reranker(enabled: bool, model: str | None):
    if not enabled:
        return None
    from .rerank import DEFAULT_RERANK_MODEL, load_reranker
    return load_reranker(model or DEFAULT_RERANK_MODEL,
                         progress=lambda m: print(m, file=sys.stderr))


def _parse_thresholds(spec: str) -> dict:
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition("=")
        out[key.strip()] = float(val)
    return out


def cmd_eval(args) -> int:
    from .eval import (build_name_dataset, build_scaffold, load_curated,
                       evaluate, format_report)
    import json as _json
    repo = Path(args.repo)

    if args.scaffold:
        for row in build_scaffold(repo, max_items=args.max_items):
            print(_json.dumps(row))
        return 0

    dataset = []
    if args.curated:
        dataset += load_curated(Path(args.curated))
    if args.kind in ("name", "both") or not args.curated:
        dataset += build_name_dataset(repo, max_items=args.max_items)
    if not dataset:
        raise SystemExit("Empty eval dataset.")

    reranker = _reranker(args.rerank, args.rerank_model)
    report = evaluate(repo, dataset, _embedder(args.model), k=args.k,
                      reranker=reranker, progress=lambda m: print(m, file=sys.stderr))
    if args.json:
        print(_json.dumps(report))
    else:
        print(format_report(report))

    # CI gate: fail the build if the gated mode drops below any threshold.
    if args.fail_under:
        gate_mode = args.check_mode or ("hybrid+rerank" if reranker else "hybrid")
        metrics = report["modes"].get(gate_mode)
        if metrics is None:
            print(f"fail-under: mode '{gate_mode}' not in report", file=sys.stderr)
            return 2
        thresholds = _parse_thresholds(args.fail_under)
        failures = []
        for key, floor in thresholds.items():
            got = metrics.get(key)
            if got is None:
                print(f"fail-under: unknown metric '{key}' "
                      f"(have: {', '.join(metrics)})", file=sys.stderr)
                return 2
            if got < floor:
                failures.append(f"{gate_mode}.{key} = {got:.3f} < {floor:.3f}")
        if failures:
            print("QUALITY GATE FAILED:", file=sys.stderr)
            for f in failures:
                print("  " + f, file=sys.stderr)
            return 1
        print(f"quality gate passed ({gate_mode}: "
              f"{', '.join(f'{k}={metrics[k]:.3f}' for k in thresholds)})",
              file=sys.stderr)
    return 0


def cmd_index(args) -> int:
    from .index import build_index
    build_index(Path(args.repo), _embedder(args.model),
                batch_size=args.batch, force=args.full, build_graph=not args.no_graph)
    return 0


def _load_graph_or_exit(repo: str):
    from .callgraph import load_graph
    from .index import index_dir_for
    g = load_graph(index_dir_for(Path(repo)))
    if g is None:
        raise SystemExit(f"No call graph found. Run `codenav index {repo}` first.")
    return g


def _match_or_report(g, symbol: str):
    """Resolve a symbol name to definitions; print guidance if 0 or many."""
    matches = g.find(symbol)
    if not matches:
        print(f"No definition named '{symbol}' found in the graph.")
        return None
    if len(matches) > 1:
        print(f"'{symbol}' matches {len(matches)} definitions "
              f"(showing results for all — qualify to narrow):")
        for m in matches:
            print(f"  {m.qualified}  {m.locator()}")
        print()
    return matches


def cmd_defs(args) -> int:
    g = _load_graph_or_exit(args.repo)
    matches = g.find(args.symbol)
    if args.json:
        print(json.dumps([m.to_dict() for m in matches]))
        return 0
    if not matches:
        print(f"No definition named '{args.symbol}' found.")
        return 0
    print(f"\n{len(matches)} definition(s) of '{args.symbol}':")
    for m in matches:
        print(f"  {m.qualified}  {m.locator()}  ({m.kind})")
    print()
    return 0


def cmd_callers(args) -> int:
    g = _load_graph_or_exit(args.repo)
    matches = _match_or_report(g, args.symbol) if not args.json else g.find(args.symbol)
    if not matches:
        if args.json:
            print(json.dumps([]))
        return 0
    seen, rows = set(), []
    for tgt in matches:
        for caller_id, path, line in g.callers(tgt.id):
            key = (caller_id, path, line)
            if key in seen:
                continue
            seen.add(key)
            caller = g.symbols[caller_id].qualified if caller_id >= 0 else "<module>"
            rows.append({"caller": caller, "path": path, "line": line,
                         "target": tgt.qualified})
    rows.sort(key=lambda r: (r["path"], r["line"]))
    if args.json:
        print(json.dumps(rows))
        return 0
    print(f"{len(rows)} call site(s) reaching '{args.symbol}':")
    for r in rows:
        print(f"  {r['caller']:<40} {r['path']}:{r['line']}")
    print()
    return 0


def cmd_callees(args) -> int:
    g = _load_graph_or_exit(args.repo)
    matches = _match_or_report(g, args.symbol) if not args.json else g.find(args.symbol)
    if not matches:
        if args.json:
            print(json.dumps([]))
        return 0
    internal, external = [], []
    for tgt in matches:
        for e in g.callees(tgt.id):
            if e.resolved:
                for rid in e.resolved:
                    internal.append({"callee": g.symbols[rid].qualified,
                                     "path": e.path, "line": e.line,
                                     "target_locator": g.symbols[rid].locator()})
            else:
                external.append({"callee": e.callee, "path": e.path, "line": e.line})
    if args.json:
        print(json.dumps({"internal": internal, "external": external}))
        return 0
    print(f"\n'{args.symbol}' calls {len(internal)} internal + {len(external)} external:")
    for r in internal:
        print(f"  → {r['callee']:<38} ({r['target_locator']})  @{r['path']}:{r['line']}")
    if external:
        names = sorted({r["callee"] for r in external})
        print(f"  external: {', '.join(names)}")
    print()
    return 0


def cmd_ask(args) -> int:
    from .query import retrieve
    from .llm import answer as llm_answer
    hits = retrieve(Path(args.repo), args.question, _embedder(args.model),
                    k=args.k, mode=args.mode,
                    reranker=_reranker(not args.no_rerank, args.rerank_model),
                    graph_expand=not args.no_expand)
    if args.json:
        ans = llm_answer(args.question, hits)
        print(json.dumps({"answer": ans, "hits": [h.to_dict() for h in hits]}))
    else:
        print("\n" + llm_answer(args.question, hits) + "\n")
    return 0


def cmd_search(args) -> int:
    from .query import retrieve
    hits = retrieve(Path(args.repo), args.question, _embedder(args.model),
                    k=args.k, mode=args.mode,
                    reranker=_reranker(not args.no_rerank, args.rerank_model))
    if args.json:
        print(json.dumps([h.to_dict() for h in hits]))
    else:
        body = "\n\n".join(f"[{h.score:.4f}] {h.locator()}\n{h.text}" for h in hits)
        print("\n" + body + "\n")
    return 0


def _add_retrieval_flags(p) -> None:
    p.add_argument("repo")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=8, help="number of chunks to return")
    p.add_argument("--mode", choices=("hybrid", "vector", "lexical"), default="hybrid",
                   help="stage-1 retrieval mode (default: hybrid)")
    p.add_argument("--no-rerank", action="store_true",
                   help="skip the cross-encoder re-ranking stage")
    p.add_argument("--rerank-model", dest="rerank_model", default=None,
                   help="cross-encoder model name")
    p.add_argument("--json", action="store_true", help="emit JSON")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codenav", description="Ask your codebase in plain language.")
    p.add_argument("--model", help="embedding model name (default BAAI/bge-small-en-v1.5)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="build/refresh the index for a repo")
    pi.add_argument("repo")
    pi.add_argument("--batch", type=int, default=64)
    pi.add_argument("--full", action="store_true",
                    help="rebuild from scratch instead of incrementally")
    pi.add_argument("--no-graph", action="store_true",
                    help="skip building the call graph")
    pi.set_defaults(func=cmd_index)

    pe = sub.add_parser("eval", help="measure retrieval quality across modes")
    pe.add_argument("repo")
    pe.add_argument("--kind", choices=("name", "curated", "both"), default="name",
                    help="auto name->code dataset, curated JSONL, or both")
    pe.add_argument("--curated", help="path to a curated JSONL eval set")
    pe.add_argument("--max-items", dest="max_items", type=int, default=200)
    pe.add_argument("-k", type=int, default=10)
    pe.add_argument("--rerank", action="store_true",
                    help="also evaluate the hybrid+rerank mode")
    pe.add_argument("--rerank-model", dest="rerank_model", default=None)
    pe.add_argument("--scaffold", action="store_true",
                    help="emit a curated-JSONL template from the repo's symbols and exit")
    pe.add_argument("--fail-under", dest="fail_under", default=None,
                    help="CI gate: e.g. 'recall@10=0.8,mrr=0.5' — exit 1 if below")
    pe.add_argument("--check-mode", dest="check_mode", default=None,
                    help="which mode the --fail-under thresholds apply to")
    pe.add_argument("--json", action="store_true")
    pe.set_defaults(func=cmd_eval)

    for verb, fn, helptext in (
        ("defs", cmd_defs, "where a symbol is defined"),
        ("callers", cmd_callers, "what calls a symbol"),
        ("callees", cmd_callees, "what a symbol calls"),
    ):
        pg = sub.add_parser(verb, help=helptext)
        pg.add_argument("repo")
        pg.add_argument("symbol")
        pg.add_argument("--json", action="store_true", help="emit JSON")
        pg.set_defaults(func=fn)

    pa = sub.add_parser("ask", help="ask a question (retrieval + Claude)")
    _add_retrieval_flags(pa)
    pa.add_argument("--no-expand", action="store_true",
                    help="don't pull in call-graph-linked code as extra context")
    pa.set_defaults(func=cmd_ask)

    ps = sub.add_parser("search", help="retrieval only, no LLM (no API key needed)")
    _add_retrieval_flags(ps)
    ps.set_defaults(func=cmd_search)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
