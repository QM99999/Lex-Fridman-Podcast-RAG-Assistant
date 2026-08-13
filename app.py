"""Streamlit UI for the Lex Fridman podcast RAG assistant (src/05_rag.py).

Run:
    streamlit run app.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
rag = importlib.import_module("05_rag")
import ui_config

st.set_page_config(page_title="Lex Fridman Podcast Q&A", page_icon="🎙️", layout="wide")
st.title("🎙️ Lex Fridman Podcast Q&A")
st.caption("Knowledge base: #491 Peter Steinberger · #494 Jensen Huang ｜ Hybrid retrieval (BM25 + vector, RRF fusion)")

EPISODES = [
    {
        "episode": 491, "guest": "Peter Steinberger",
        "title": "OpenClaw – the viral AI agent that broke the internet",
        "url": "https://lexfridman.com/peter-steinberger-transcript/",
    },
    {
        "episode": 494, "guest": "Jensen Huang",
        "title": "NVIDIA – the $4 trillion company & the AI revolution",
        "url": "https://lexfridman.com/jensen-huang-transcript/",
    },
]

STEP_LABELS = {
    "loading": "Loading knowledge base…",
    "rewriting": "Rewriting your question…",
    "retrieving": "Retrieving context (embedding + hybrid search)…",
    "generating": "Generating answer…",
}

ui_config.render_keys()
ui_config.render_admin_sidebar()

with st.sidebar:
    st.header("⚙️ Settings")
    default_backend = os.environ.get("APP_BACKEND", "elasticsearch")
    backend = st.selectbox(
        "Retrieval backend", ["elasticsearch", "memory"],
        index=0 if default_backend == "elasticsearch" else 1,
        help="elasticsearch requires the Dockerized ES (index built by 03); memory loads data/index/kb_memory.json directly "
             "(set APP_BACKEND=memory on Streamlit Cloud)")
    rerank = st.checkbox(
        "Rerank results (cross-encoder)", value=False,
        help="hybrid fetches more candidates, then BAAI/bge-reranker-base re-ranks them (needs: pip install fastembed)")
    rerank_model = "BAAI/bge-reranker-base" if rerank else "" 
    rewrite_q = st.checkbox(
        "Rewrite query with LLM (gpt-4.1-nano)", value=True,
        help="rewrite the question before retrieval for better matching "
             "(costs one small LLM call per question)")
    top_k = st.slider("Context chunks (Top-K)", 1, 10, 5)
    model = st.text_input(
        "Answer model", value=os.environ.get("05_MODEL", "gpt-5.4-mini"),
        help="Model that generates the answer (default: 05_MODEL from .env, the 06 bake-off winner)")
    st.caption("Feedback is saved to data/results/feedback.jsonl")

    st.divider()
    st.subheader("📄 Podcast transcripts")
    st.caption("Read the original episodes before asking:")
    for ep in EPISODES:
        st.markdown(f"**#{ep['episode']} · {ep['guest']}** – {ep['title']}")
        st.markdown(f"[{ep['url']}]({ep['url']})")


@st.cache_resource
def get_pipeline(backend: str, rerank_model: str):
    return rag.build_pipeline(backend=backend, rerank_model=rerank_model)


st.subheader("Ask a question")
question = st.text_input(
    "Your question",
    placeholder="e.g. What does Jensen think about AGI?",
    key="question",
)
ask = st.button("Ask", type="primary")

if ask and question.strip():
    try:
        with st.status("Working…", expanded=True) as status:
            status.update(label=STEP_LABELS["loading"], state="running")
            index, embed, reranker = get_pipeline(backend, rerank_model)
            status.write(f"Knowledge base loaded (backend: {backend})")
            status.update(label=STEP_LABELS["retrieving"], state="running")
            result = rag.answer(
                question.strip(), backend=backend, top_k=top_k,
                model=model.strip() or None, index=index, embed=embed,
                reranker=reranker,
                rewrite_model="gpt-4.1-nano" if rewrite_q else None,
                step_cb=lambda s: status.update(label=STEP_LABELS.get(s, s)),
            )
            status.update(label="Done", state="complete")
    except Exception as exc:  # noqa: BLE001 - surface any API/index error in the UI
        st.error(f"Something went wrong: {exc}")
        st.session_state.pop("result", None)
    else:
        st.session_state["result"] = result
        st.session_state["asked_question"] = question.strip()

result = st.session_state.get("result")
if result and st.session_state.get("asked_question") == question.strip():
    if result.get("error"):
        st.warning(f"No relevant context found: {result['error']}")
    else:
        meta = f"retrieval {result['retrieval_ms']} ms · {result['provider']}/{result['model']}"
        if result.get("usage"):
            u = result["usage"]
            meta += f" · tokens: {u.get('total_tokens')}"
        st.caption(meta)
        st.markdown(result["answer"])

        with st.expander(f"📚 Source chunks ({len(result['docs'])})"):
            for i, d in enumerate(result["docs"], 1):
                st.markdown(
                    f"**[{i}]** Episode {d['episode']} · {d['guest']} · "
                    f"Chapter \"{d['chapter']}\" · {d['start_ts']}-{d['end_ts']}")
                st.text(d["text"])

        col1, col2 = st.columns(2)
        if col1.button("👍 Helpful"):
            rag.save_feedback({
                "rating": 1, "question": question.strip(), "answer": result["raw"],
                "model": result["model"], "doc_ids": [d["id"] for d in result["docs"]],
            })
            st.success("Feedback saved 👍, thank you!")
        if col2.button("👎 Not helpful"):
            rag.save_feedback({
                "rating": -1, "question": question.strip(), "answer": result["raw"],
                "model": result["model"], "doc_ids": [d["id"] for d in result["docs"]],
            })
            st.success("Feedback saved 👎, thank you!")
elif not result:
    st.info("Type a question and click Ask.")
