"""Step 4 - Evaluate retrieval approaches (BM25 / vector / hybrid).

Uses the FAQ dataset as golden queries: each FAQ question is searched with the
three approaches; ground truth chunks are those whose time range contains the
FAQ's source_timestamps (mapped against the chunk artifact written by
03_build_index.py). Reports hit_rate@k, MRR and recall@k per approach.

Incremental & resumable: per-query results are checkpointed to the results file
after every query. Re-running skips queries that already have results (matched
by faq_id), so interrupted runs resume where they left off and newly added FAQ
items are evaluated incrementally. Metrics are always recomputed from the stored
per-query details; the checkpoint is invalidated when backend / depth / ks /
chunk count change (pass --force to start fresh).

Usage:
    python src/04_retrieval_eval.py --backend memory
    python src/04_retrieval_eval.py --backend elasticsearch
    python src/04_retrieval_eval.py --ks 1,3,5 --depth 10 --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from retrieval import (PROJECT_ROOT, ElasticIndex, MemoryIndex, Reranker,
                       load_faq_queries, make_embedder, map_timestamps, rrf)

CHECKPOINT_EVERY = 25


def _compute_metrics(details: list[dict], ks: list[int], depth: int,
                     methods: tuple[str, ...] = ("bm25", "vector", "hybrid")) -> tuple[dict, int, int]:
    with_gt = [d for d in details if d["gt"]]
    n = len(with_gt)
    no_gt = len(details) - n
    agg = {m: {"hits": {k: 0 for k in ks}, "rr": 0.0,
               "recall_num": 0, "recall_den": 0} for m in methods}
    for d in with_gt:
        gt_set = set(d["gt"])
        for m in methods:
            ids = d["ranked"].get(m, [])
            for k in ks:
                if gt_set & set(ids[:k]):
                    agg[m]["hits"][k] += 1
            for pos, doc_id in enumerate(ids, 1):
                if doc_id in gt_set:
                    agg[m]["rr"] += 1.0 / pos
                    break
            agg[m]["recall_num"] += len(gt_set & set(ids))
            agg[m]["recall_den"] += len(gt_set)
    metrics = {
        m: {"hit_rate": {k: agg[m]["hits"][k] / n for k in ks} if n else {k: 0.0 for k in ks},
            "mrr": agg[m]["rr"] / n if n else 0.0,
            "recall": agg[m]["recall_num"] / agg[m]["recall_den"] if agg[m]["recall_den"] else 0.0}
        for m in methods
    }
    return metrics, n, no_gt


def default_results_name(backend: str, depth: int, per: int, ks: list[int],
                        limit: int, rerank_model: str | None) -> str:
    """Auto-named results file so different configs never overwrite each other."""
    b = "es" if backend == "elasticsearch" else "mem"
    name = f"retrieval_eval_{b}_d{depth}_p{per}_k{'-'.join(map(str, ks))}"
    if rerank_model:
        name += "_rerank-" + rerank_model.replace("/", "-")
    if limit:
        name += f"_limit{limit}"
    return name + ".json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval approaches (incremental + resumable).")
    parser.add_argument("--backend", choices=["memory", "elasticsearch"], default="memory")
    parser.add_argument("--faq", default=str(PROJECT_ROOT / "data/faq"))
    parser.add_argument("--artifact", default=str(PROJECT_ROOT / "data/index" / "kb_memory.json"))
    parser.add_argument("--es-url", default=None, help="default: ES_URL env or http://localhost:9200")
    parser.add_argument("--index-name", default="lex_fridman")
    parser.add_argument("--depth", type=int, default=10, help="retrieval depth used for ranking")
    parser.add_argument("--per", type=int, default=None,
                        help="candidates per method (BM25/vector) before RRF fusion (default: depth)")
    parser.add_argument("--ks", default="1,3,5", help="comma-separated k values for hit_rate@k")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N queries (0 = all)")
    parser.add_argument("--rerank-model", default=None,
                        help="cross-encoder model to re-rank hybrid candidates (e.g. BAAI/bge-reranker-base); empty = off")
    parser.add_argument("--rerank-candidates", type=int, default=20,
                        help="hybrid candidates fetched before reranking (default: 20)")
    parser.add_argument("--results", default="",
                        help="results file (empty = auto-named by backend/depth/ks/per)")
    parser.add_argument("--force", action="store_true", help="ignore the results checkpoint and start fresh")
    args = parser.parse_args()

    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    chunks = artifact["docs"]
    print(f"artifact: {len(chunks)} chunks (embedding_model={artifact.get('embedding_model')})")

    if args.backend == "memory":
        index = MemoryIndex()
        docs_for_bm25 = [{k: v for k, v in d.items() if k != "vector"} for d in chunks]
        index.add_docs(docs_for_bm25)
        index.add_vectors([d["vector"] for d in chunks])
        print("backend: memory (pure-Python BM25 + numpy cosine)")
    else:
        es_url = args.es_url or os.environ.get("ES_URL", "http://localhost:9200")
        index = ElasticIndex(es_url, args.index_name)
        if not index.exists():
            raise SystemExit(f"index '{args.index_name}' not found @ {es_url} - run 03_build_index.py first")
        print(f"backend: elasticsearch {args.index_name} @ {es_url}")

    queries = load_faq_queries(args.faq)
    if args.limit:
        queries = queries[: args.limit]
    ks = [int(x) for x in args.ks.split(",")]
    per = args.per or args.depth
    methods = ("bm25", "vector", "hybrid") + (("hybrid+rerank",) if args.rerank_model else ())
    reranker = None
    if args.rerank_model:
        reranker = Reranker(args.rerank_model)
        print(f"reranker: {args.rerank_model} (candidates={args.rerank_candidates})")

    out_path = Path(args.results) if args.results else \
        PROJECT_ROOT / "data/results" / default_results_name(
            args.backend, args.depth, per, ks, args.limit, args.rerank_model)
    results: dict = {}
    details: list[dict] = []
    if out_path.exists() and not args.force:
        results = json.loads(out_path.read_text(encoding="utf-8"))
        details = results.get("details", [])
        rerank_on = bool(args.rerank_model)
        compatible = (results.get("backend") == args.backend
                      and results.get("depth") == args.depth
                      and results.get("ks") == ks
                      and results.get("num_chunks") == len(chunks)
                      and (results.get("per") or results.get("depth")) == per
                      and results.get("rerank_model") == args.rerank_model
                      and results.get("rerank_candidates") == args.rerank_candidates)
        if compatible and rerank_on:
            compatible = all("hybrid+rerank" in d.get("ranked", {}) for d in details)
        if not compatible:
            results = {}
            details = []
            print("existing checkpoint incompatible (backend/depth/ks/chunks changed) - starting fresh")
    done_ids = {d["faq_id"] for d in details if d.get("faq_id")}
    todo = [q for q in queries if q["faq_id"] not in done_ids]
    print(f"golden queries: {len(queries)} total, {len(details)} cached, {len(todo)} to evaluate")

    if todo:
        embed = make_embedder()
        by_episode: dict = {}
        for c in chunks:
            by_episode.setdefault(c["episode"], []).append(c)

        for qi, q in enumerate(todo, 1):
            gt = map_timestamps(by_episode.get(q["episode"], []), q["timestamps"])
            ranked = {"bm25": [], "vector": [], "hybrid": []}
            if gt:
                try:
                    vec = embed([q["question"]])[0]
                    bm = index.search_bm25(q["question"], per)
                    vec_top = index.search_vector(vec, per)
                    ranked = {
                        "bm25": [doc_id for doc_id, _ in bm[: args.depth]],
                        "vector": [doc_id for doc_id, _ in vec_top[: args.depth]],
                        "hybrid": [doc_id for doc_id, _ in rrf([bm, vec_top], top=args.depth)],
                    }
                    if reranker is not None:
                        candidates = rrf([bm, vec_top], top=args.rerank_candidates)
                        ranked["hybrid+rerank"] = [
                            doc_id for doc_id, _ in reranker.rerank(
                                q["question"], candidates, chunks, top=args.depth)
                        ]
                except Exception as exc:
                    print(f"  !! query {q['faq_id']} failed: {exc} (will be retried on next run)")
            details.append({
                "faq_id": q["faq_id"],
                "episode": q["episode"],
                "question": q["question"],
                "gt": gt,
                "ranked": ranked,
            })
            if len(details) % CHECKPOINT_EVERY == 0:
                metrics, n, no_gt = _compute_metrics(details, ks, args.depth, methods)
                _save(out_path, args, ks, len(chunks), len(queries), n, no_gt, len(details),
                      metrics, details, failed=0)
                print(f"  checkpoint: {len(details)}/{len(queries)} evaluated")
        print(f"done evaluating {len(todo)} new queries")

    metrics, n, no_gt = _compute_metrics(details, ks, args.depth, methods)
    _save(out_path, args, ks, len(chunks), len(queries), n, no_gt, len(details),
          metrics, details, failed=0)

    print(f"saved results -> {out_path}")

    print(f"\nretrieval evaluation on {n} queries (k={ks}, depth={args.depth}):")
    print(f"{'method':<8} " + "  ".join(f"hit@{k}" for k in ks) +
          f"  {'MRR':>6}  {'recall@' + str(args.depth):>10}")
    for m in methods:
        mm = metrics[m]
        hit_cols = "  ".join(f"{mm['hit_rate'][k]:6.3f}" for k in ks)
        print(f"{m:<8} {hit_cols}  {mm['mrr']:6.3f}  {mm['recall']:10.3f}")
    if no_gt:
        print(f"({no_gt} queries had no mappable ground truth)")
    print(f"\nsaved results -> {out_path}")


def _save(out_path: Path, args, ks, num_chunks, num_queries, with_gt, no_gt,
          evaluated, metrics, details, failed: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "backend": args.backend,
        "depth": args.depth,
        "ks": ks,
        "num_chunks": num_chunks,
        "num_queries_total": num_queries,
        "num_queries_evaluated": evaluated,
        "num_queries_with_gt": with_gt,
        "num_queries_no_gt": no_gt,
        "num_queries_failed": failed,
        "per": args.per or args.depth,
        "rerank_model": args.rerank_model,
        "rerank_candidates": args.rerank_candidates,
        "metrics": metrics,
        "details": details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
