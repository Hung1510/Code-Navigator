#!/usr/bin/env python
"""
calibrate.py — the run that turns CodeNavigator's quality claims into numbers.

Two jobs, in order:

  1. ABLATE the instruction-prefix scheme. embedder.py can prefix queries and
     passages three ways (bge / e5 / none) and the right one is a measurable
     question, not a matter of taste. Each scheme needs its own index, because
     the passage prefix changes the document vectors — so we force a full
     re-index per scheme. This is the expensive part.

  2. CALIBRATE the CI gate. Once we know the best scheme, we read the observed
     scores and propose `--fail-under` floors that sit below normal noise but
     above a real regression.

On the noise margin: the eval is deterministic for a fixed index, so there is
no run-to-run variance to worry about. The uncertainty that matters is
*sampling* uncertainty — your dataset is n queries drawn from a much larger
space of questions people might ask. Treating recall as a binomial proportion,
the standard error is sqrt(p*(1-p)/n). At n=82 and p=0.7 that's ~0.05, so a
floor 2 standard errors below the observed score (~10pp) is the honest choice.
We compute it rather than eyeball it.

Usage:
    python scripts/calibrate.py <repo> [<repo> ...] \
        [--curated REPO=path.jsonl] [--rerank] [-k 10] [--out results.json]

The first run downloads BGE-small (~130MB) and, with --rerank, the
cross-encoder (~80MB). Both are cached afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from codenavigator.embedder import PREFIX_SCHEMES, FastEmbedder
from codenavigator.eval import build_name_dataset, evaluate, load_curated
from codenavigator.index import build_index
from codenavigator.rerank import load_reranker

QUIET = lambda *a, **k: None


def stderr_margin(p: float, n: int, sigmas: float = 2.0) -> float:
    """Binomial standard error of a proportion, scaled. Guard n=0."""
    if n <= 0:
        return 0.0
    return sigmas * math.sqrt(max(p * (1.0 - p), 1e-9) / n)


def run_one(repo: Path, prefix: str, k: int, max_items: int,
            curated: Path | None, use_rerank: bool) -> dict:
    """Full re-index under one prefix scheme, then score every mode."""
    embedder = FastEmbedder(prefix=prefix)
    t0 = time.time()
    build_index(repo, embedder, force=True, progress=QUIET)
    index_s = time.time() - t0

    dataset = build_name_dataset(repo, max_items=max_items)
    if curated:
        dataset += load_curated(curated)

    reranker = load_reranker() if use_rerank else None
    t0 = time.time()
    report = evaluate(repo, dataset, embedder, k=k, reranker=reranker, progress=QUIET)
    report["prefix"] = prefix
    report["index_seconds"] = round(index_s, 1)
    report["eval_seconds"] = round(time.time() - t0, 1)
    report["curated_items"] = sum(1 for i in dataset if i.source == "curated")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+", type=Path)
    ap.add_argument("--curated", action="append", default=[],
                    metavar="REPO=PATH", help="attach a curated JSONL to a repo, "
                                              "e.g. --curated sla=evals/sla.jsonl")
    ap.add_argument("--prefixes", default="bge,e5,none",
                    help="prefix schemes to ablate (default: all three)")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--rerank", action="store_true",
                    help="also score hybrid+rerank (downloads the cross-encoder)")
    ap.add_argument("--gate-repo", type=Path, default=None,
                    help="which repo the CI gate runs on (default: first repo)")
    ap.add_argument("--gate-mode", default="hybrid")
    ap.add_argument("--out", type=Path, default=Path("calibration.json"))
    args = ap.parse_args()

    curated_map: dict[str, Path] = {}
    for spec in args.curated:
        name, _, path = spec.partition("=")
        curated_map[name] = Path(path)

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    for p in prefixes:
        if p not in PREFIX_SCHEMES:
            raise SystemExit(f"unknown prefix {p!r}; choose from {sorted(PREFIX_SCHEMES)}")

    results: list[dict] = []
    for repo in args.repos:
        repo = repo.resolve()
        curated = curated_map.get(repo.name)
        for prefix in prefixes:
            print(f"[{repo.name}] prefix={prefix} — indexing + scoring ...", flush=True)
            r = run_one(repo, prefix, args.k, args.max_items, curated, args.rerank)
            r["repo"] = repo.name
            results.append(r)
            print(f"    n={r['n']} (curated {r['curated_items']}) "
                  f"index {r['index_seconds']}s / eval {r['eval_seconds']}s")

    # ---- report -----------------------------------------------------------
    kk = f"recall@{args.k}"
    print(f"\n{'repo':<26}{'prefix':<8}{'mode':<16}{'r@1':>8}{kk:>10}{'MRR':>8}")
    print("-" * 76)
    for r in results:
        for mode, m in r["modes"].items():
            print(f"{r['repo']:<26}{r['prefix']:<8}{mode:<16}"
                  f"{m['recall@1']:>8.3f}{m[kk]:>10.3f}{m['mrr']:>8.3f}")

    # ---- which prefix wins? ----------------------------------------------
    # Judge on the vector mode: it's the only one the prefix can affect.
    # (lexical is prefix-blind; hybrid dilutes the signal with BM25.)
    print("\nPrefix ablation — vector mode only (the modes BM25 can't rescue):")
    for repo_name in dict.fromkeys(r["repo"] for r in results):
        rows = [r for r in results if r["repo"] == repo_name]
        best = max(rows, key=lambda r: r["modes"]["vector"]["mrr"])
        for r in rows:
            v = r["modes"]["vector"]
            flag = "  <-- best" if r is best else ""
            print(f"  {repo_name:<26}{r['prefix']:<8}"
                  f"MRR {v['mrr']:.3f}  r@1 {v['recall@1']:.3f}{flag}")

    # ---- CI thresholds ----------------------------------------------------
    gate_repo = (args.gate_repo or args.repos[0]).resolve().name
    gate_rows = [r for r in results if r["repo"] == gate_repo]
    if gate_rows:
        best = max(gate_rows, key=lambda r: r["modes"]["vector"]["mrr"])
        m = best["modes"].get(args.gate_mode)
        if m:
            n = best["n"]
            rk, mrr = m[kk], m["mrr"]
            floor_rk = max(0.0, rk - stderr_margin(rk, n))
            floor_mrr = max(0.0, mrr - stderr_margin(mrr, n))
            print(f"\nCI gate — repo={gate_repo} mode={args.gate_mode} "
                  f"prefix={best['prefix']} n={n}")
            print(f"  observed  {kk}={rk:.3f}  mrr={mrr:.3f}")
            print(f"  2-sigma   ±{stderr_margin(rk, n):.3f} / ±{stderr_margin(mrr, n):.3f}")
            print(f"  suggested --fail-under \"{kk}={floor_rk:.2f},mrr={floor_mrr:.2f}\"")
            if n < 100:
                print(f"  ! n={n} is small; the floors above are wide. Consider a "
                      f"bigger fixture corpus.")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
