"""Pipeline page - run the 01-06 pipeline from the UI.

Collects the same options as src/07_pipeline.py in a form, then runs the
Prefect flow in a subprocess and streams its output live, so you never need
to type commands or edit code to run (or re-run) any stage.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE = PROJECT_ROOT / "src" / "07_pipeline.py"
LOG_FILE = PROJECT_ROOT / "data/results" / "pipeline_run.log"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
import ui_config


def log_tail(max_chars: int = 8000) -> str:
    """Tail of the live pipeline log; tqdm frames are collapsed to the last one."""
    if not LOG_FILE.exists():
        return "(no log yet - start a run first)"
    text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    lines = text.replace("\r", "\n").splitlines()
    filtered, last_progress = [], None
    for ln in lines:
        # tqdm updates one line per frame; drop all but the latest frame.
        # Speed may be shown as "it/s" or "s/it" depending on pace.
        if "%|" in ln and ("it/s" in ln or "s/it" in ln):
            last_progress = ln
        else:
            filtered.append(ln)
    if last_progress is not None:
        filtered.append(last_progress)
    return "\n".join(filtered)[-max_chars:]


def current_step() -> str:
    """Last step marker printed by 07 (e.g. "[2/2 rag_eval]")."""
    if not LOG_FILE.exists():
        return "(starting...)"
    text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    marks = re.findall(r"\[(\d+/\d+ [a-z_]+)\]", text)
    return marks[-1] if marks else "(starting...)"

st.set_page_config(page_title="Pipeline", page_icon="\u2699\ufe0f", layout="wide")
if not ui_config.require_admin():
    st.stop()

st.title("\u2699\ufe0f Pipeline")
st.caption("One-command orchestration of steps 01-06 (Prefect flow). "
           "Rerank, backend, evaluation options and more are set here - no code edits needed.")

st.subheader(":heavy_plus_sign: Add a new Lex episode")
with st.expander("Paste a Lex Fridman URL to download its transcript", expanded=False):
    c1, c2 = st.columns([3, 1])
    ep_url = c1.text_input(
        "Lex Fridman transcript page URL",
        placeholder="https://lexfridman.com/<guest>-transcript/",
        help="Only episodes with an official transcript page are supported "
             "(e.g. https://lexfridman.com/jensen-huang-transcript/).",
        key="ep_url",
    )
    if c2.button("Download transcript", type="secondary"):
        if not ep_url.strip():
            st.error("Paste a Lex Fridman URL first.")
        else:
            fetch_cmd = [sys.executable, str(PROJECT_ROOT / "src" / "00_fetch_transcript.py"),
                         "--url", ep_url.strip()]
            st.code(" ".join(fetch_cmd), language="bash")
            with st.spinner("Downloading transcript from lexfridman.com..."):
                try:
                    r = subprocess.run(fetch_cmd, cwd=PROJECT_ROOT, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace",
                                       timeout=120)
                    st.code((r.stdout or "") + (r.stderr or ""), language="text")
                    if r.returncode == 0:
                        st.success("Transcript saved to data/raw. Next: select steps clean -> faq -> index "
                                   "and run the pipeline (faq generates Q&A for the new episode, costs LLM tokens).")
                    else:
                        st.error("Download failed - see output above.")
                except subprocess.TimeoutExpired:
                    st.error("Download timed out after 120s - try again or check the URL.")

ui_config.render_keys()

with st.sidebar:
    st.header("\u2699\ufe0f Steps")
    steps = st.multiselect(
        "Select steps (in order)",
        ["clean", "faq", "index", "retrieval_eval", "rag_eval"],
        default=[],
        help="faq extracts Q&A pairs with an LLM (costs tokens on new episodes); "
             "re-running fills in only new/incomplete episodes unless Force is on",
    )
    default_backend = os.environ.get("APP_BACKEND", "elasticsearch")
    backend = st.selectbox("Backend", ["elasticsearch", "memory"],
                           index=0 if default_backend == "elasticsearch" else 1,
                           help="elasticsearch = Docker ES (production); memory = local kb_memory.json "
                                "(set APP_BACKEND=memory on Streamlit Cloud)")
    rerank = st.checkbox("Rerank results", value=False,
                         help="hybrid fetches more candidates, then bge-reranker re-ranks them")
    rerank_model = "BAAI/bge-reranker-base" if rerank else ""
    force = st.checkbox("Force (ignore checkpoints)", value=False)

ALL_STEPS = ["clean", "faq", "index", "retrieval_eval", "rag_eval"]


def steps_flow(selected: list[str]) -> str:
    """HTML pipeline chain: selected steps highlighted, others greyed out."""
    nodes = []
    for step in ALL_STEPS:
        on = step in selected
        bg = "#2e7d32" if on else "#e9ecef"
        fg = "#ffffff" if on else "#6c757d"
        weight = "bold" if on else "normal"
        nodes.append(
            f'<span style="background:{bg};color:{fg};padding:3px 14px;'
            f'border-radius:14px;font-weight:{weight}">{step}</span>'
        )
    arrow = '<span style="color:#adb5bd;padding:0 4px">&#10132;</span>'
    return arrow.join(nodes)


st.markdown("**Pipeline flow**")
if steps:
    st.markdown(steps_flow(steps), unsafe_allow_html=True)
else:
    st.caption("Select steps on the left to see the execution flow.")

with st.form("pipeline_form"):
    if "faq" in steps:
        st.markdown("**FAQ generation (02)**")
        faq_model = st.text_input(
            "02 model (FAQ generation)",
            value=os.environ.get("02_MODEL", "gpt-4.1-mini"),
            key="faq_model",
            help="model used to extract Q&A pairs from transcripts (saved to .env)")
    if "retrieval_eval" in steps:
        st.markdown("**Retrieval evaluation (04)**")
        c1, c2 = st.columns(2)
        per = c1.number_input("per (0 = depth)", min_value=0, max_value=200, value=0,
                              help="candidates per method before RRF fusion", key="per")
        depth = c2.number_input("depth", min_value=1, max_value=50, value=10, key="depth")
    if "rag_eval" in steps:
        st.markdown("**RAG evaluation (06)**")
        c3, c4 = st.columns(2)
        top_k = c3.number_input("top_k (06)", min_value=1, max_value=20, value=5, key="top_k")
        sample = c4.number_input("06 sample (0 = all)", min_value=0, max_value=287, value=0, key="sample")
        answer_models = st.text_input("06 answer models (comma)",
                                      value="gpt-3.5-turbo,gpt-4o-mini,gpt-5.4-mini",
                                      help="athletes to evaluate (default: the three GPT bake-off models)",
                                      key="answer_models")
        c5, c6 = st.columns(2)
        judge_model = c5.text_input("06 judge model", value="gpt-5.6-luna", key="judge_model")
        rewrite_model = c6.text_input("06 rewrite model", value="gpt-4.1-nano", key="rewrite_model")
        rewrite_q = st.checkbox("Rewrite query (LLM)", value=True,
                                help="rewrite questions before retrieval; rewriter must NOT be an answer model",
                                key="rewrite_q")
        out_file = st.text_input("06 results file (blank = auto)",
                                 value="",
                                 help="blank = auto-named by config (e.g. rag_eval_es_k5_rw-gpt-4.1-nano_judge-...json), "
                                      "so different configs never overwrite each other",
                                 key="out_file")
    run = st.form_submit_button("Run pipeline", type="primary")

if run:
    if not steps:
        st.error("Select at least one step.")
        st.stop()
    cmd = [sys.executable, str(PIPELINE), "--steps", ",".join(steps), "--backend", backend]
    if rerank:
        cmd += ["--rerank-model", rerank_model]
    if force:
        cmd += ["--force"]
    if "faq" in steps:
        cmd += ["--include-faq"]
        if faq_model.strip():
            ui_config.write_env("02_MODEL", faq_model.strip())
    if "retrieval_eval" in steps:
        if per:
            cmd += ["--per", str(per)]
        cmd += ["--depth", str(depth)]
    if "rag_eval" in steps:
        cmd += ["--top-k", str(top_k)]
        if answer_models.strip():
            cmd += ["--answer-models", answer_models.strip()]
        if judge_model.strip():
            cmd += ["--judge-model", judge_model.strip()]
        cmd += ["--rewrite", "llm" if rewrite_q else "none"]
        if rewrite_model.strip():
            cmd += ["--rewrite-model", rewrite_model.strip()]
        if out_file.strip():
            cmd += ["--out", out_file.strip()]
        if sample:
            cmd += ["--sample", str(sample)]

    st.code(" ".join(cmd), language="bash")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, stdout=logf,
            stderr=subprocess.STDOUT, env=env,
        )
    st.session_state.pipeline_proc = proc
    st.rerun()

@st.fragment(run_every=1)
def pipeline_status():
    """Auto-refreshing status block: polls the background process every second,
    so progress updates and the finished state show up without clicks."""
    proc = st.session_state.get("pipeline_proc")
    if proc is None:
        return
    status = proc.poll()
    if status is None:
        st.info(f"Pipeline is running in the background (PID {proc.pid}). "
                "Switching pages won't stop it - this block refreshes itself.")
        st.markdown(f"**Current step: {current_step()}**")
        with st.expander("Live log", expanded=True):
            st.code(log_tail())
        if st.button("Abort pipeline"):
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 - process may already be gone
                pass
            st.session_state.pipeline_proc = None
            st.rerun()
    else:
        st.success(f"Pipeline finished (exit code {status}).")
        with st.expander("Log", expanded=True):
            st.code(log_tail())
        if st.button("Clear result"):
            st.session_state.pipeline_proc = None
            st.rerun()


pipeline_status()
