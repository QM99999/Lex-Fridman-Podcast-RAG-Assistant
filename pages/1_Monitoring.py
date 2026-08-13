"""Monitoring dashboard for the Lex Fridman podcast RAG assistant.

Shown as the "Monitoring" page when running `streamlit run app.py`.

Reads:
    data/results/retrieval_eval.json  - retrieval benchmark (BM25 / vector / hybrid)
    data/results/feedback.jsonl       - thumbs up/down feedback from the Q&A page
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "how",
    "what", "why", "who", "when", "where", "which", "of", "to", "in", "on",
    "for", "with", "about", "it", "he", "she", "his", "her", "its", "and",
    "or", "not", "you", "your", "i", "we", "they", "that", "this", "has",
    "have", "had", "can", "could", "would", "should", "will", "from", "as",
    "at", "by", "be", "been", "being", "than", "then", "there", "their",
}

st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import ui_config
if not ui_config.require_admin():
    st.stop()

st.title("📊 Monitoring")

# ---------------------------------------------------------------- feedback
st.subheader("User feedback")
fb_path = PROJECT_ROOT / "data/results" / "feedback.jsonl"
records = []
if fb_path.exists():
    for line in fb_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

if not records:
    st.info("No feedback collected yet. Use the 👍/👎 buttons on the Q&A page.")
else:
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df["label"] = df["rating"].map({1: "helpful", -1: "not helpful"})
    total = len(df)
    good = int((df["rating"] == 1).sum())
    bad = total - good

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Feedback", total)
    m2.metric("👍 helpful", good)
    m3.metric("👎 not helpful", bad)
    m4.metric("Positive rate", f"{good / total:.1%}")

    st.markdown("**Recent questions (last 5)**")
    recent = df.dropna(subset=["ts"]).sort_values("ts", ascending=False).head(5)
    if not recent.empty:
        recent_tbl = pd.DataFrame({
            "time (UTC)": recent["ts"].dt.strftime("%Y-%m-%d %H:%M"),
            "question": recent["question"].astype(str).str.slice(0, 120),
            "model": recent["model"].fillna(""),
            "rating": recent["rating"].map({1: "helpful", -1: "not helpful", 0: "no rating"}),
        })
        st.dataframe(recent_tbl, use_container_width=True, hide_index=True)


    st.markdown("**👍 / 👎 counts**")
    st.bar_chart(df["label"].value_counts())

    ts = df.dropna(subset=["ts"]).set_index("ts")
    if not ts.empty:
        st.markdown("**Feedback over time (daily)**")
        # Aggregate by calendar day in the local timezone (Europe/Berlin),
        # so a late-evening UTC timestamp counts toward the correct day.
        local = ts.index.tz_convert("Europe/Berlin").tz_localize(None)
        daily = local.to_series(index=local).resample("1D").size()
        daily_df = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in daily.index],
            "count": daily.values,
        })
        st.bar_chart(daily_df, x="date", y="count")

    cited_eps: Counter = Counter()
    for cid_list in df.get("doc_ids", pd.Series(dtype=object)):
        for cid in cid_list or []:
            ep = str(cid).split("-")[0]
            if ep:
                cited_eps[ep] += 1
    if cited_eps:
        st.markdown("**Chunks cited by episode**")
        st.bar_chart(pd.Series(dict(cited_eps)).sort_index())

    st.markdown("**Top words in user questions**")
    words: Counter = Counter()
    for q in df["question"].dropna():
        words.update(w for w in re.findall(r"[a-z0-9']+", q.lower()) if w not in STOPWORDS)
    if words:
        st.bar_chart(pd.Series(dict(words.most_common(15))))

    if "answer" in df and df["answer"].notna().any():
        st.markdown("**Answer length distribution (words)**")
        lens = df["answer"].dropna().astype(str).str.split().str.len()
        if not lens.empty:
            bins = [0, 50, 100, 150, 200, 300, 500, 1000]
            labels = ["<50", "50-100", "100-150", "150-200", "200-300", "300-500", "500+"]
            dist = pd.cut(lens, bins=bins, labels=labels, right=False).value_counts().sort_index()
            st.bar_chart(dist)

    if "retrieval_ms" in df and df["retrieval_ms"].notna().any():
        st.markdown("**Retrieval latency (ms)**")
        st.line_chart(df.set_index("ts")["retrieval_ms"].dropna())

st.caption("Sources: `data/results/feedback.jsonl`")
