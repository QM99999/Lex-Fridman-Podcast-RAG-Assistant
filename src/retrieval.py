"""Shared retrieval layer: chunking, embedding, search backends, RRF fusion.

Used by:
    * 03_build_index.py     - build the knowledge base (chunks + embeddings)
    * 04_retrieval_eval.py  - evaluate BM25 / vector / hybrid retrieval

The search backend is swappable via --backend:
    memory          in-memory numpy + pure-Python Okapi BM25 (no server)
    elasticsearch   Elasticsearch (docker) with BM25 + dense kNN

Chunk design: chunks are built per chapter, capped at --max-words words, and
dialogue segments are never split. Each chunk stores its start/end timestamps so
FAQ source_timestamps can be mapped to the chunk(s) containing them - that
mapping is the ground truth for retrieval evaluation.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def load_dotenv(path: Path) -> None:
    """Tiny .env loader (no extra dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_dotenv(PROJECT_ROOT / ".env")


def ts_to_seconds(ts: str) -> int:
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def load_episodes(processed_dir: str) -> list[dict]:
    folder = Path(processed_dir)
    files = sorted(folder.glob("*.json"))
    if not files:
        raise SystemExit(f"No cleaned episode files found in {folder}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _chunk_text(segments: list[dict]) -> str:
    return "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)


def _make_chunk(episode: dict, chapter: dict, ci: int, segments: list[dict],
                start_ts: str, end_ts: str, n: int) -> dict:
    return {
        "id": f"{episode.get('episode')}-{ci:02d}-{n:03d}",
        "episode": episode.get("episode"),
        "guest": episode.get("guest"),
        "chapter": chapter["title"],
        "chapter_index": ci,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "text": _chunk_text(segments),
    }


def chunk_episode(episode: dict, max_words: int = 500) -> list[dict]:
    """Split one episode into chunks: per chapter, never splitting a segment."""
    chunks: list[dict] = []
    n = 0
    for ci, chapter in enumerate(episode.get("chapters", []), 1):
        segments = chapter.get("segments", [])
        if not segments:
            continue
        cur, cur_words, start_ts = [], 0, None
        for seg in segments:
            w = len(seg["text"].split())
            if cur and cur_words + w > max_words:
                n += 1
                chunks.append(_make_chunk(episode, chapter, ci, cur, start_ts,
                                          cur[-1]["timestamp"], n))
                cur, cur_words, start_ts = [], 0, None
            if not cur:
                start_ts = seg["timestamp"]
            cur.append(seg)
            cur_words += w
        if cur:
            n += 1
            chunks.append(_make_chunk(episode, chapter, ci, cur, start_ts,
                                      cur[-1]["timestamp"], n))
    return chunks


def chunk_all(episodes: list[dict], max_words: int = 500) -> list[dict]:
    return [c for ep in episodes for c in chunk_episode(ep, max_words)]


def make_embedder(model: str | None = None, api_key: str | None = None,
                  base_url: str | None = None):
    """Return a function embedding a list of texts into a list of vectors."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    model = model or os.environ.get("03_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors.extend(d.embedding for d in ordered)
        return vectors

    embed.model = model
    return embed


TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Pure-Python Okapi BM25 (k1=1.5, b=0.75), built once from documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[dict] = []
        self.doc_terms: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.df: dict[str, int] = {}
        self.N = 0

    def build(self, docs: list[dict]) -> None:
        self.docs = docs
        self.doc_terms = []
        self.doc_len = []
        self.df = {}
        for doc in docs:
            terms = tokenize(doc["text"])
            self.doc_terms.append(terms)
            self.doc_len.append(len(terms))
            for term in set(terms):
                self.df[term] = self.df.get(term, 0) + 1
        self.N = len(docs)
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        dl = self.doc_len[doc_idx]
        score = 0.0
        for term in query_terms:
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            tf = self.doc_terms[doc_idx].count(term)
            if not tf:
                continue
            tf_norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            score += idf * tf_norm
        return score

    def search(self, query: str, top: int = 10) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        scores = [(self.docs[i]["id"], self._score(query_terms, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s for s in scores if s[1] > 0][:top]


def rrf(results: list[list[tuple[str, float]]], k: int = 60, top: int = 10) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over several ranked id lists."""
    fused: dict[str, float] = {}
    for ranked in results:
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ordered[:top]


class Reranker:
    """Local cross-encoder reranker (bge-reranker-base, ONNX Runtime, no torch).

    Scores (query, chunk) pairs and re-ranks a candidate list, so the final
    top-k is picked by semantic relevance instead of BM25/vector similarity
    alone. The ONNX weights (Xenova/bge-reranker-base, quantized ~280MB) are
    downloaded once on first use and cached locally; everything runs on CPU.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-base"   # logical name (mapping to ONNX)
    ONNX_REPO = "Xenova/bge-reranker-base"
    MAX_LENGTH = 512

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []

    # ONNX weight variants (smallest first; int8 ~279MB is fine for ranking).
    WEIGHT_VARIANTS = ("model_int8.onnx", "model_quantized.onnx", "model.onnx")

    @staticmethod
    def local_dir() -> Path:
        """Where the model files live if downloaded manually (bypasses HF hub)."""
        override = os.environ.get("RERANK_MODEL_DIR")
        base = Path(override) if override else Path.home() / ".cache" / "lex-rag-models"
        return base / "bge-reranker-base"

    def _ensure(self):
        if self._session is not None:
            return
        try:
            import numpy as np  # noqa: F401  (kept imported for callers)
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "reranking needs onnxruntime + tokenizers - "
                "run: pip install onnxruntime tokenizers"
            ) from exc

        # 1) local files (manually downloaded, e.g. from hf-mirror)
        local = self.local_dir()
        onnx_path = next((local / v for v in self.WEIGHT_VARIANTS
                          if (local / v).exists()), None)
        tok_path = local / "tokenizer.json"
        if onnx_path is None or not tok_path.exists():
            onnx_path = tok_path = None
        # 2) HuggingFace hub (official endpoint or HF_ENDPOINT mirror)
        if onnx_path is None:
            try:
                from huggingface_hub import hf_hub_download
                for variant in self.WEIGHT_VARIANTS:
                    try:
                        onnx_path = Path(hf_hub_download(
                            self.ONNX_REPO, f"onnx/{variant}"))
                        break
                    except Exception:  # noqa: BLE001 - try the next variant
                        continue
                if onnx_path is not None:
                    tok_path = Path(hf_hub_download(self.ONNX_REPO, "tokenizer.json"))
            except ImportError:
                pass
        if onnx_path is None or tok_path is None:
            raise RuntimeError(
                "rerank model not found. Either:\n"
                f"  1) put model_int8.onnx + tokenizer.json into: {local}\n"
                "  (download e.g. from https://hf-mirror.com/Xenova/bge-reranker-base)\n"
                "  or 2) set HF_ENDPOINT to a reachable hub and retry."
            )
        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"])
        self._input_names = [i.name for i in self._session.get_inputs()]
        tok = Tokenizer.from_file(str(tok_path))
        tok.enable_truncation(max_length=self.MAX_LENGTH)
        tok.enable_padding(pad_id=1, pad_token="<pad>")
        self._tokenizer = tok

    def rerank(self, query: str, candidates: list[tuple[str, float]],
               docs: list[dict], top: int = 5) -> list[tuple[str, float]]:
        """Score (query, chunk.text) pairs and return the top ids as (id, score)."""
        import numpy as np
        self._ensure()
        by_id = {d["id"]: d for d in docs}
        ids = [cid for cid, _ in candidates if cid in by_id]
        texts = [by_id[cid]["text"] for cid in ids]
        if not ids:
            return []
        encs = self._tokenizer.encode_batch([(query, text) for text in texts])
        feeds = {
            "input_ids": np.asarray([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encs], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
        logits = self._session.run(None, feeds)[0].reshape(-1)
        pairs = sorted(zip(ids, logits.tolist()), key=lambda p: p[1], reverse=True)
        return [(i, float(s)) for i, s in pairs[:top]]


class MemoryIndex:
    """In-memory backend: pure-Python BM25 + numpy cosine vector search."""

    def __init__(self):
        self.docs: list[dict] = []
        self.ids: list[str] = []
        self.vectors = None
        self.bm25 = BM25Index()

    def add_docs(self, docs: list[dict]) -> None:
        self.docs = docs
        self.ids = [d["id"] for d in docs]
        self.bm25.build(docs)

    def add_vectors(self, vectors: list[list[float]]) -> None:
        import numpy as np
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search_bm25(self, query: str, top: int = 10) -> list[tuple[str, float]]:
        return self.bm25.search(query, top)

    def get_docs(self, ids: list[str]) -> list[dict]:
        by_id = {d["id"]: d for d in self.docs}
        return [by_id[i] for i in ids if i in by_id]

    def search_vector(self, query_vec: list[float], top: int = 10) -> list[tuple[str, float]]:
        import numpy as np
        q = np.asarray(query_vec, dtype=np.float32)
        norms = np.linalg.norm(self.vectors, axis=1)
        scores = self.vectors @ q / (norms * np.linalg.norm(q) + 1e-9)
        order = np.argsort(scores)[::-1][:top]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in order]

    def hybrid(self, query: str, query_vec: list[float], top: int = 10,
               rrf_k: int = 60, per: int = 10) -> list[tuple[str, float]]:
        return rrf([self.search_bm25(query, per), self.search_vector(query_vec, per)],
                   k=rrf_k, top=top)


class ElasticIndex:
    """Elasticsearch backend: BM25 (match query) + dense kNN; hybrid fused via RRF."""

    def __init__(self, url: str, index_name: str):
        from elasticsearch import Elasticsearch
        self.client = Elasticsearch(url)
        self.index_name = index_name

    def exists(self) -> bool:
        return self.client.indices.exists(index=self.index_name)

    def create_index(self, dims: int, force: bool = False) -> None:
        if self.exists():
            if not force:
                return
            self.client.indices.delete(index=self.index_name)
        self.client.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "id": {"type": "keyword"},
                    "episode": {"type": "integer"},
                    "guest": {"type": "keyword"},
                    "chapter": {"type": "keyword"},
                    "chapter_index": {"type": "integer"},
                    "start_ts": {"type": "keyword"},
                    "end_ts": {"type": "keyword"},
                    "text": {"type": "text"},
                    "vector": {"type": "dense_vector", "dims": dims,
                               "index": True, "similarity": "cosine"},
                }
            },
        )

    def add_docs(self, docs_with_vectors: list[dict]) -> None:
        from elasticsearch.helpers import bulk
        actions = [
            {"_index": self.index_name, "_id": d["id"], "_source": d}
            for d in docs_with_vectors
        ]
        bulk(self.client, actions)

    def _hits(self, body: dict, top: int) -> list[tuple[str, float]]:
        resp = self.client.search(index=self.index_name, body=body, size=top)
        return [(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]

    def search_bm25(self, query: str, top: int = 10) -> list[tuple[str, float]]:
        return self._hits({"query": {"match": {"text": query}}}, top)

    def get_docs(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        resp = self.client.mget(index=self.index_name, body={"ids": ids})
        docs = []
        for d in resp["docs"]:
            if d.get("found"):
                s = dict(d["_source"])
                s.pop("vector", None)  # keep the payload light
                docs.append(s)
        return docs

    def search_vector(self, query_vec: list[float], top: int = 10) -> list[tuple[str, float]]:
        return self._hits(
            {"knn": {"field": "vector", "query_vector": query_vec, "k": top,
                     "num_candidates": 200}},
            top,
        )

    def hybrid(self, query: str, query_vec: list[float], top: int = 10,
               rrf_k: int = 60, per: int = 10) -> list[tuple[str, float]]:
        return rrf([self.search_bm25(query, per), self.search_vector(query_vec, per)],
                   k=rrf_k, top=top)


def load_faq_queries(faq_dir: str) -> list[dict]:
    """Load FAQ items as golden queries: question + episode + source timestamps."""
    queries: list[dict] = []
    for f in sorted(Path(faq_dir).glob("*.faq.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for qa in data.get("faqs", []):
            queries.append({
                "faq_id": qa.get("id"),
                "question": qa["question"],
                "episode": data.get("episode"),
                "timestamps": qa.get("source_timestamps", []),
            })
    return queries


def map_timestamps(chunks: list[dict], timestamps: list[str]) -> list[str]:
    """Chunk ids whose [start_ts, end_ts] contains any of the timestamps."""
    hits: set[str] = set()
    for ts in timestamps:
        sec = ts_to_seconds(ts)
        for c in chunks:
            if ts_to_seconds(c["start_ts"]) <= sec <= ts_to_seconds(c["end_ts"]):
                hits.add(c["id"])
                break
    return sorted(hits)

REWRITE_SYSTEM_PROMPT = (
    "You rewrite questions for a podcast transcript RAG system. "
    "Rewrite the user's question into ONE clear, self-contained, "
    "search-friendly question. Expand pronouns and abbreviations, "
    "mention the episode guest/topic when obvious, keep the original "
    "meaning, stay in English. Output ONLY the rewritten question."
)


def rewrite_query(question: str, model: str, client, max_retries: int = 2) -> str:
    """Rewrite a user question into a retrieval-friendly query (single query).

    Used before hybrid retrieval so BM25/vector matching sees a clear,
    self-contained question instead of pronouns or vague wording. The
    original question is still what gets answered - rewriting only feeds
    the search.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.2,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as exc:  # noqa: BLE001 - retry on transient failures
            last_exc = exc
    raise RuntimeError(f"query rewrite failed after {max_retries} attempts: {last_exc}")
