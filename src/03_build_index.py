"""Step 3 - Build the retrieval knowledge base (chunks + embeddings).

Reads cleaned episodes from data/processed, chunks them (per chapter,
--max-words words, no segment splitting), embeds the chunk text with OpenAI,
and writes the result to the chosen backend:

    --backend memory          writes data/index/kb_memory.json (chunks + vectors)
    --backend elasticsearch   indexes into Elasticsearch (docker)

Incremental & resumable: the artifact (chunks + vectors) is checkpointed after
every embedding batch. Re-running only embeds chunks that are not already in the
artifact, so interrupted runs resume where they left off and newly added
episodes only cost embeddings for their new chunks. The ES index is upserted by
doc id, so re-indexing is idempotent.

Caveat: changing --max-words or the embedding model changes chunk boundaries /
vectors, so the artifact is rebuilt instead of resumed in that case.

Configuration:
    03_EMBEDDING_MODEL  embedding model (default: text-embedding-3-small)
    OPENAI_API_KEY      required
    ES_URL              Elasticsearch URL (default: http://localhost:9200)

Usage:
    python src/03_build_index.py --backend memory
    python src/03_build_index.py --backend elasticsearch --force   # rebuild ES index
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from retrieval import (PROJECT_ROOT, ElasticIndex, chunk_all, load_episodes,
                       make_embedder)

EMBED_BATCH = 32


def _save_artifact(path: Path, max_words: int, model: str, docs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"max_words": max_words, "embedding_model": model, "docs": docs},
                   ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the retrieval knowledge base (incremental + resumable).")
    parser.add_argument("--backend", choices=["memory", "elasticsearch"], default="memory")
    parser.add_argument("--max-words", type=int, default=500, help="chunk size in words")
    parser.add_argument("--processed", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--artifact", default=str(PROJECT_ROOT / "data/index" / "kb_memory.json"))
    parser.add_argument("--es-url", default=None, help="default: ES_URL env or http://localhost:9200")
    parser.add_argument("--index-name", default="lex_fridman")
    parser.add_argument("--force", action="store_true", help="ignore the artifact and re-embed everything")
    args = parser.parse_args()

    episodes = load_episodes(args.processed)
    chunks = chunk_all(episodes, args.max_words)
    print(f"episodes={len(episodes)} chunks={len(chunks)} "
          f"(max_words={args.max_words}, per-chapter, no segment splitting)")

    embed = make_embedder()
    artifact = Path(args.artifact)
    docs: list[dict] = []
    if artifact.exists() and not args.force:
        saved = json.loads(artifact.read_text(encoding="utf-8"))
        if saved.get("max_words") == args.max_words and saved.get("embedding_model") == embed.model:
            docs = saved["docs"]
    existing = {d["id"] for d in docs}
    todo = [c for c in chunks if c["id"] not in existing]
    if docs:
        print(f"artifact: {len(docs)} docs cached ({embed.model}); new chunks to embed: {len(todo)}")
    else:
        print(f"no usable artifact; embedding all {len(todo)} chunks")

    if todo:
        for i in range(0, len(todo), EMBED_BATCH):
            batch = todo[i : i + EMBED_BATCH]
            try:
                vectors = embed([c["text"] for c in batch])
            except Exception as exc:
                _save_artifact(artifact, args.max_words, embed.model, docs)
                raise SystemExit(
                    f"embedding batch {i // EMBED_BATCH + 1} failed: {exc}\n"
                    f"progress checkpointed ({len(docs)}/{len(chunks)} docs) - re-run to resume"
                )
            for c, vec in zip(batch, vectors):
                d = dict(c)
                d["vector"] = vec
                docs.append(d)
            _save_artifact(artifact, args.max_words, embed.model, docs)
            print(f"  embedded {len(docs)}/{len(chunks)} chunks")
        print(f"saved artifact -> {artifact}")

    if args.backend == "elasticsearch":
        es_url = args.es_url or os.environ.get("ES_URL", "http://localhost:9200")
        index = ElasticIndex(es_url, args.index_name)
        index.create_index(dims=len(docs[0]["vector"]), force=args.force)
        index.add_docs(docs)  # upsert by doc id: idempotent / incremental-safe
        print(f"indexed {len(docs)} docs (upsert) -> {args.index_name} @ {es_url}")
    else:
        print(f"memory artifact ready: {artifact} ({len(docs)} docs)")


if __name__ == "__main__":
    main()
