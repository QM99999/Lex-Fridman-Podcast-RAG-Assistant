"""Step 5 - RAG question answering over the knowledge base.

Flow: question -> hybrid retrieval (BM25 + vector, RRF fusion) -> top-k
chunks -> LLM answer grounded in the chunks, with inline [n] citations and an
appended Sources section (episode, guest, chapter, timestamps).

Answers are always generated in English.

Also exposes the pieces reused by the Streamlit app (app.py):
    build_pipeline()  build (index, embedder) once, reuse across questions
    answer()          end-to-end: retrieve + generate
    save_feedback()   thumbs up/down records appended to feedback.jsonl

Configuration (env / .env):
    05_MODEL            answer model (default: gpt-5.4-mini)
    OPENAI_API_KEY      required when provider=openai
    DEEPSEEK_API_KEY    required when provider=deepseek
    ES_URL              Elasticsearch URL (default: http://localhost:9200)
    03_EMBEDDING_MODEL  embedding model (used by retrieval.make_embedder)

Usage:
    python src/05_rag.py "What does Jensen think about AGI safety?"
    python src/05_rag.py "How does Jensen view AI safety?" --backend memory --top-k 3
    python src/05_rag.py --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from retrieval import (PROJECT_ROOT, ElasticIndex, MemoryIndex, Reranker,
                       make_embedder, rewrite_query)

PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url_default": "https://api.openai.com/v1",
        "model_env": "OPENAI_MODEL",
        "model_default": "gpt-5.4-mini",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url_default": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_MODEL",
        "model_default": "deepseek-v4-flash",
    },
}

DEFAULT_ARTIFACT = PROJECT_ROOT / "data" / "index" / "kb_memory.json"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "results" / "feedback.jsonl"

SYSTEM_PROMPT = (
    "You are a question-answering assistant for Lex Fridman podcast "
    "transcripts (episode #491 with Peter Steinberger and episode #494 with "
    "Jensen Huang). Answer ONLY from the provided context chunks. Cite chunks "
    "inline as [n] right after the sentence that uses them. If the context does "
    "not contain the answer, say so clearly instead of guessing. Do not add a "
    "sources section at the end - the inline citations are enough."
)


def resolve_provider(cli_provider: str | None, model_env: str) -> str:
    """Decide the provider: CLI arg > inferred from model name > openai."""
    if cli_provider:
        return cli_provider
    if "deepseek" in model_env.lower():
        return "deepseek"
    return "openai"


_RERANKER_CACHE: dict[str, Reranker] = {}


def build_reranker(model: str | None) -> Reranker | None:
    """Return a cached Reranker instance (None when reranking is off)."""
    if not model:
        return None
    if model not in _RERANKER_CACHE:
        _RERANKER_CACHE[model] = Reranker(model)
    return _RERANKER_CACHE[model]


def build_pipeline(backend: str = "memory", artifact: str | None = None,
                   es_url: str | None = None, index_name: str = "lex_fridman",
                   rerank_model: str | None = None):
    """Build a reusable (index, embedder, reranker).

    memory        loads data/index/kb_memory.json (chunks + vectors)
    elasticsearch talks to the dockerized ES index built by 03_build_index.py
    rerank_model  cross-encoder model to re-rank candidates (empty = off)
    """
    embed = make_embedder()
    if backend == "memory":
        path = Path(artifact or DEFAULT_ARTIFACT)
        if not path.exists():
            raise RuntimeError(
                f"artifact not found: {path} - run src/03_build_index.py --backend memory first"
            )
        art = json.loads(path.read_text(encoding="utf-8"))
        chunks = art["docs"]
        index = MemoryIndex()
        index.add_docs([{k: v for k, v in d.items() if k != "vector"} for d in chunks])
        index.add_vectors([d["vector"] for d in chunks])
        return index, embed, build_reranker(rerank_model)
    url = es_url or os.environ.get("ES_URL", "http://localhost:9200")
    index = ElasticIndex(url, index_name)
    if not index.exists():
        raise RuntimeError(
            f"index '{index_name}' not found @ {url} - start docker (ES) and "
            "run src/03_build_index.py --backend elasticsearch first"
        )
    return index, embed, build_reranker(rerank_model)


def retrieve(index, embed, question: str, top_k: int = 5, per: int = 10,
             reranker=None, rerank_candidates: int = 20) -> list[dict]:
    """Hybrid search (BM25 + vector, RRF); optionally re-rank the candidates
    with a cross-encoder before taking the final top_k."""
    vec = embed([question])[0]
    top = max(top_k, rerank_candidates) if reranker is not None else top_k
    ranked = index.hybrid(question, vec, top=top, per=per)
    if not ranked:
        return []
    if reranker is not None:
        docs_all = index.get_docs([doc_id for doc_id, _ in ranked])
        ranked = reranker.rerank(question, ranked, docs_all, top=top_k)
    scores = dict(ranked)
    docs = index.get_docs([doc_id for doc_id, _ in ranked])
    for d in docs:
        d["score"] = scores.get(d["id"], 0.0)
    return docs


def _source_line(d: dict, n: int) -> str:
    return (f"[{n}] (Episode {d['episode']} · {d['guest']} · "
            f"chapter \"{d['chapter']}\" · {d['start_ts']}-{d['end_ts']})\n{d['text']}")


def build_messages(question: str, docs: list[dict]) -> list[dict]:
    context = "\n\n".join(_source_line(d, i + 1) for i, d in enumerate(docs))
    user = f"Question: {question}\n\nContext chunks:\n{context}"
    user += "\n\nAnswer in English."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def format_answer(raw: str, docs: list[dict]) -> tuple[str, list[str]]:
    """Append a Sources section listing the cited chunk sources (or all chunks)."""
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", raw)})
    lines: list[str] = []
    for n in cited:
        if 0 < n <= len(docs):
            d = docs[n - 1]
            lines.append(f"[{n}] Episode {d['episode']} · {d['guest']} · "
                         f"chapter \"{d['chapter']}\" · {d['start_ts']}-{d['end_ts']}")
    if not lines:
        for i, d in enumerate(docs, 1):
            lines.append(f"[{i}] Episode {d['episode']} · {d['guest']} · "
                         f"chapter \"{d['chapter']}\" · {d['start_ts']}-{d['end_ts']}")
    if re.search(r"(?i)source", raw):
        return raw, lines
    return raw + "\n\nSources:\n" + "\n".join(lines), lines


def answer(question: str, *, backend: str = "memory", top_k: int = 5,
           per: int = 10, provider: str | None = None, model: str | None = None,
           temperature: float = 0.2, timeout: float = 120,
           index=None, embed=None, reranker=None, rerank_candidates: int = 20,
           rerank_model: str | None = None, client=None, step_cb=None,
           rewrite_model: str | None = None, rewrite_client=None,
           retrieve_query: str | None = None) -> dict:
    """End-to-end RAG answer with sources.

    Pass pre-built index/embed (and optionally client) to reuse them across
    questions, e.g. from the Streamlit app. `step_cb` (optional) is called
    with a phase name at each step: "loading" (only when building the
    pipeline), "retrieving", "generating".
    """
    t0 = time.time()
    if index is None or embed is None:
        if step_cb:
            step_cb("loading")
        index, embed, reranker = build_pipeline(backend, rerank_model=rerank_model)
    if retrieve_query is None:
        retrieve_query = question
    if rewrite_model:
        if step_cb:
            step_cb("rewriting")
        rc = rewrite_client
        if rc is None:
            from openai import OpenAI
            rk = os.environ.get("OPENAI_API_KEY", "")
            if not rk:
                raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env")
            rc = OpenAI(api_key=rk,
                        base_url=os.environ.get("OPENAI_BASE_URL",
                                                "https://api.openai.com/v1"),
                        timeout=timeout)
        retrieve_query = rewrite_query(question, rewrite_model, rc)
    if step_cb:
        step_cb("retrieving")
    docs = retrieve(index, embed, retrieve_query, top_k, per,
                    reranker=reranker, rerank_candidates=rerank_candidates)
    retrieval_ms = round((time.time() - t0) * 1000)
    if not docs:
        return {
            "question": question, "answer": "", "raw": "",
            "sources": [], "docs": [],
            "provider": None, "model": None, "usage": None,
            "retrieval_ms": retrieval_ms,
            "error": "no relevant context found",
        }

    model_env = model or os.environ.get("05_MODEL", "")
    provider = resolve_provider(provider, model_env)
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"{cfg['api_key_env']} is not set. Add it to .env")
    model = model_env or os.environ.get(cfg["model_env"], cfg["model_default"])
    base_url = os.environ.get(cfg["base_url_env"], cfg["base_url_default"])
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    if step_cb:
        step_cb("generating")
    messages = build_messages(question, docs)
    resp = client.chat.completions.create(model=model, messages=messages,
                                          temperature=temperature)
    raw = (resp.choices[0].message.content or "").strip()
    final, source_lines = format_answer(raw, docs)
    return {
        "question": question,
        "retrieve_query": retrieve_query,
        "answer": final,
        "raw": raw,
        "sources": source_lines,
        "docs": docs,
        "provider": provider,
        "model": model,
        "usage": resp.usage.model_dump() if resp.usage else None,
        "retrieval_ms": retrieval_ms,
        "error": None,
    }


def save_feedback(entry: dict, path: str | Path | None = None) -> Path:
    """Append one feedback record (one JSON line) to data/results/feedback.jsonl."""
    out = Path(path) if path else FEEDBACK_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    record = dict(entry)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out


def load_feedback(path: str | Path | None = None) -> list[dict]:
    out = Path(path) if path else FEEDBACK_PATH
    if not out.exists():
        return []
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _print_result(result: dict) -> None:
    print(f"Q: {result['question']}")
    meta = [f"retrieval {result['retrieval_ms']}ms",
            f"provider={result['provider']}", f"model={result['model']}"]
    if result.get("usage"):
        u = result["usage"]
        meta.append(f"tokens={u.get('total_tokens')} "
                    f"(prompt {u.get('prompt_tokens')} / completion {u.get('completion_tokens')})")
    print(f"[{' | '.join(meta)}]")
    print()
    print(result["answer"])
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG question answering over the podcast knowledge base.")
    parser.add_argument("question", nargs="?", default=None, help="question to answer")
    parser.add_argument("--backend", choices=["memory", "elasticsearch"], default="memory")
    parser.add_argument("--top-k", type=int, default=5, help="number of context chunks (default: 5)")
    parser.add_argument("--per", type=int, default=10, help="per-method candidates before RRF fusion")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=None,
                        help="openai | deepseek (default: inferred from 05_MODEL, then openai)")
    parser.add_argument("--model", default=None, help="override 05_MODEL / provider default")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--rerank-model", default=None,
                        help="cross-encoder model to re-rank candidates (e.g. BAAI/bge-reranker-base); empty = off")
    parser.add_argument("--rerank-candidates", type=int, default=20,
                        help="candidates fetched before reranking (default: 20)")
    parser.add_argument("--rewrite-model", default=None,
                        help="LLM model to rewrite the question before retrieval "
                             "(e.g. gpt-4.1-nano); empty = off")
    parser.add_argument("--interactive", action="store_true", help="chat loop")
    args = parser.parse_args()

    try:
        index, embed, reranker = build_pipeline(args.backend, rerank_model=args.rerank_model)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}")

    if args.interactive or not args.question:
        print("Lex Fridman Q&A (Ctrl+C or type 'exit' to quit)")
        while True:
            try:
                q = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            try:
                result = answer(q, backend=args.backend, top_k=args.top_k,
                                per=args.per, provider=args.provider,
                                model=args.model, timeout=args.timeout,
                                index=index, embed=embed, reranker=reranker,
                                rerank_candidates=args.rerank_candidates,
                                rewrite_model=args.rewrite_model)
            except RuntimeError as exc:
                print(f"error: {exc}\n")
                continue
            if result.get("error"):
                print(f"error: {result['error']}\n")
                continue
            _print_result(result)
        return

    result = answer(args.question, backend=args.backend, top_k=args.top_k,
                    per=args.per, provider=args.provider, model=args.model,
                    timeout=args.timeout, index=index, embed=embed,
                    reranker=reranker, rerank_candidates=args.rerank_candidates,
                    rewrite_model=args.rewrite_model)
    if result.get("error"):
        raise SystemExit(f"error: {result['error']}")
    _print_result(result)


if __name__ == "__main__":
    main()
