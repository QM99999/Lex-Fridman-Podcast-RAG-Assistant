"""Step 7 - One-command pipeline orchestrator (Prefect flow).

Chains steps 01-06 into a single parameterized flow:

    clean -> faq -> index -> retrieval_eval -> rag_eval

Run any subset with --steps; every option is passed through to the underlying
scripts, so you never need to edit code to change settings (backend, rerank
on/off, per, force, 06 answer models, ...).

The flow is plain Prefect: run it directly (python src/07_pipeline.py ...) or
from the Streamlit "Pipeline" page. No Prefect server is required for local
runs - task logs go to the console.

Usage:
    python src/07_pipeline.py --steps index,retrieval_eval
    python src/07_pipeline.py --steps index,retrieval_eval --backend elasticsearch --rerank-model BAAI/bge-reranker-base
    python src/07_pipeline.py --steps clean,index,retrieval_eval,rag_eval --force
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Prefect spins a throwaway sqlite server per run; its telemetry heartbeat
# logs "database is locked" noise inside containers. Disable it.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from prefect import flow, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_STEPS = ("clean", "faq", "index", "retrieval_eval", "rag_eval")


@task
def run_script(script: str, args: list[str], step_label: str = "") -> str:
    """Run one src/<script> subprocess; raise on non-zero exit."""
    cmd = [sys.executable, str(PROJECT_ROOT / "src" / script), *args]
    prefix = f"[{step_label}] " if step_label else ""
    print(f"{prefix}$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{script} failed (exit {proc.returncode})")
    return script


@flow
def pipeline(
    steps: str = "index,retrieval_eval",
    backend: str = "memory",
    force: bool = False,
    rerank_model: str = "",
    rerank_candidates: int = 20,
    per: int = 0,          # 0 = depth (per-method candidates before RRF)
    depth: int = 10,
    ks: str = "1,3,5",
    limit: int = 0,        # 0 = all (04 only)
    faq_skip: bool = True,  # FAQ benchmark already generated; set False to regenerate (costs LLM)
    top_k: int = 5,
    answer_models: str = "",
    judge_model: str = "",
    rewrite: str = "llm",  # none | llm (06 only)
    rewrite_model: str = "gpt-4.1-nano",
    out: str = "",         # 06: results file (empty = auto-named by 06 config)
    sample: int = 0,       # 0 = all (06 only)
    seed: int = 42,
    workers: int = 6,
):
    selected = [s.strip() for s in steps.split(",") if s.strip()]
    invalid = set(selected) - set(ALL_STEPS)
    if invalid:
        raise ValueError(f"unknown steps: {sorted(invalid)} (available: {ALL_STEPS})")
    if not selected:
        raise ValueError("--steps is empty")

    planned = [s for s in selected if s != "faq" or not faq_skip]
    total = len(planned)
    step_no = 0

    if "clean" in selected:
        step_no += 1
        run_script("01_clean_raw.py", [], f"{step_no}/{total} clean")

    if "faq" in selected and not faq_skip:
        step_no += 1
        faq_args = ["--force"] if force else []
        run_script("02_generate_faq.py", faq_args, f"{step_no}/{total} faq")

    if "index" in selected:
        step_no += 1
        idx_args = ["--backend", backend]
        if force:
            idx_args.append("--force")
        run_script("03_build_index.py", idx_args, f"{step_no}/{total} index")

    if "retrieval_eval" in selected:
        step_no += 1
        eval_args = ["--backend", backend, "--depth", str(depth), "--ks", ks]
        if per:
            eval_args += ["--per", str(per)]
        if rerank_model:
            eval_args += ["--rerank-model", rerank_model,
                          "--rerank-candidates", str(rerank_candidates)]
        if limit:
            eval_args += ["--limit", str(limit)]
        if force:
            eval_args.append("--force")
        run_script("04_retrieval_eval.py", eval_args, f"{step_no}/{total} retrieval_eval")

    if "rag_eval" in selected:
        step_no += 1
        rag_args = ["--backend", backend, "--top-k", str(top_k),
                    "--rewrite", rewrite]
        if out:
            rag_args += ["--out", out]
        if rewrite_model:
            rag_args += ["--rewrite-model", rewrite_model]
        if answer_models:
            rag_args += ["--answer-models", answer_models]
        if judge_model:
            rag_args += ["--judge-model", judge_model]
        if sample:
            rag_args += ["--sample", str(sample), "--seed", str(seed)]
        if rerank_model:
            rag_args += ["--rerank-model", rerank_model,
                         "--rerank-candidates", str(rerank_candidates)]
        if force:
            rag_args.append("--force")
        run_script("06_rag_eval.py", rag_args, f"{step_no}/{total} rag_eval")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command pipeline (Prefect flow).")
    parser.add_argument("--steps", default="index,retrieval_eval",
                        help="comma-separated subset of: clean,faq,index,retrieval_eval,rag_eval")
    parser.add_argument("--backend", choices=["memory", "elasticsearch"], default="memory")
    parser.add_argument("--force", action="store_true", help="ignore checkpoints / rebuild")
    parser.add_argument("--rerank-model", default="", help="cross-encoder model (empty = off)")
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--per", type=int, default=0, help="candidates per method before RRF (0 = depth)")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--ks", default="1,3,5")
    parser.add_argument("--limit", type=int, default=0, help="04: only first N queries (0 = all)")
    parser.add_argument("--include-faq", action="store_true",
                        help="regenerate the FAQ benchmark (costs LLM; off by default)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answer-models", default="", help="06: comma-separated answer models")
    parser.add_argument("--judge-model", default="", help="06: judge model")
    parser.add_argument("--rewrite", choices=["none", "llm"], default="llm",
                        help="06: rewrite questions with an LLM before retrieval (default: llm)")
    parser.add_argument("--rewrite-model", default="gpt-4.1-nano",
                        help="06: rewriter model (must not be an answer model)")
    parser.add_argument("--out", default="", help="06: results file (empty = auto-named by 06 config)")
    parser.add_argument("--sample", type=int, default=0, help="06: only N random queries (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    pipeline(
        steps=args.steps,
        backend=args.backend,
        force=args.force,
        rerank_model=args.rerank_model,
        rerank_candidates=args.rerank_candidates,
        per=args.per,
        depth=args.depth,
        ks=args.ks,
        limit=args.limit,
        faq_skip=not args.include_faq,
        top_k=args.top_k,
        answer_models=args.answer_models,
        judge_model=args.judge_model,
        rewrite=args.rewrite,
        rewrite_model=args.rewrite_model,
        out=args.out,
        sample=args.sample,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
