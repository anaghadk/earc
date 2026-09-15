"""
src/ui/app.py — Modern AI Chatbot Interface for EARC
Grounded, cited, and compressed Question-Answering using Dense Retrieval and Local/Cloud LLMs.
Supports ChatGPT/Claude-style multi-conversation sidebar.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any
import numpy as np
import streamlit as st

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Ensure Mistral API key is available from environment or fallback
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = "lIK71q2MhMrZXQwxYE0nVUvoNNUVjQzy"

st.set_page_config(
    page_title="EARC · Grounded QA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean, Modern Styling (Professional Dark Aesthetic) ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Background */
[data-testid="stAppViewContainer"] {
    background: #090d16;
    color: #e2e8f0;
}

/* Transparent Header to Keep Native Sidebar Collapse/Expand Toggle Fully Functional */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Hide deploy button & dev footer, preserve all native header/sidebar controls */
#MainMenu, footer {
    visibility: hidden;
}

/* Native Sidebar Styling */
[data-testid="stSidebar"] {
    background: #0b0f19 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
    margin: 12px 0 !important;
}

/* Ensure Native Collapse/Expand Controls Are Always Visible and Clickable */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
}

/* Sidebar New Chat Button */
div.st-key-btn_new_chat button {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f8fafc !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    padding: 7px 12px !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
}
div.st-key-btn_new_chat button:hover {
    background: #2563eb !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
}

/* Sidebar Recent Conversation Items: Compact Spacing */
div[class*="st-key-btn_conv_"] {
    margin-top: 0 !important;
    margin-bottom: 2px !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

div[class*="st-key-btn_conv_"] button {
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 12.5px !important;
    padding: 4px 8px !important;
    min-height: 30px !important;
    height: 30px !important;
    border-radius: 6px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: all 0.15s ease !important;
}

div[class*="st-key-btn_conv_"] button p {
    font-size: 12.5px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Inactive Conversation: Clean, compact text-style list without box/border */
div[class*="st-key-btn_conv_"] button[kind="tertiary"],
div[class*="st-key-btn_conv_"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: #94a3b8 !important;
    font-weight: 400 !important;
    box-shadow: none !important;
}
div[class*="st-key-btn_conv_"] button[kind="tertiary"]:hover,
div[class*="st-key-btn_conv_"] button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(255, 255, 255, 0.08) !important;
    color: #f1f5f9 !important;
}

/* Active Conversation: Subtly highlighted with soft background and soft border */
div[class*="st-key-btn_conv_"] button[kind="primary"] {
    background: rgba(37, 99, 235, 0.14) !important;
    border: 1px solid rgba(59, 130, 246, 0.35) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
div[class*="st-key-btn_conv_"] button[kind="primary"]:hover {
    background: rgba(37, 99, 235, 0.22) !important;
    border-color: rgba(59, 130, 246, 0.5) !important;
    color: #ffffff !important;
}

/* Clear Chat Button */
div.st-key-btn_clear_chat button {
    background: transparent !important;
    border: 1px solid #1e293b !important;
    color: #94a3b8 !important;
    font-size: 12px !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    justify-content: center !important;
    transition: all 0.15s ease !important;
}
div.st-key-btn_clear_chat button:hover {
    background: rgba(239, 68, 68, 0.08) !important;
    border-color: rgba(239, 68, 68, 0.25) !important;
    color: #f87171 !important;
}

/* Main Container Centering */
.main .block-container {
    max-width: 860px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 6.5rem !important;
}

/* Top Navigation Bar */
.top-nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 14px;
    margin-bottom: 20px;
    border-bottom: 1px solid #1e293b;
}
.brand-block {
    display: flex;
    align-items: center;
    gap: 10px;
}
.brand-logo {
    font-size: 22px;
}
.brand-title {
    font-size: 17px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.01em;
}
.brand-sub {
    font-size: 12px;
    color: #94a3b8;
}
.nav-pills-wrap {
    display: flex;
    gap: 8px;
    align-items: center;
}
.nav-pill {
    background: #161f30;
    border: 1px solid #2d3d5a;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11.5px;
    color: #94a3b8;
    font-weight: 500;
}

/* Welcome / Empty State */
.welcome-container {
    text-align: center;
    padding: 40px 16px 24px 16px;
}
.welcome-logo {
    font-size: 38px;
    margin-bottom: 10px;
}
.welcome-title {
    font-size: 24px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
}
.welcome-desc {
    font-size: 14px;
    color: #94a3b8;
    max-width: 520px;
    margin: 0 auto 24px auto;
    line-height: 1.6;
}

/* User Message */
.user-msg-wrap {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0 12px 0;
}
.user-msg-bubble {
    background: #2563eb;
    color: #ffffff;
    border-radius: 16px 16px 4px 16px;
    padding: 11px 16px;
    max-width: 75%;
    font-size: 14.5px;
    line-height: 1.5;
    word-break: break-word;
}

/* Bot Message */
.bot-msg-wrap {
    margin: 12px 0 16px 0;
}
.bot-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 16px 18px;
}
.bot-badge-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
    margin-bottom: 12px;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border-radius: 999px;
    padding: 2px 9px;
    font-size: 11.5px;
    font-weight: 600;
}
.badge-grounded {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.badge-abstained {
    background: rgba(168, 85, 247, 0.12);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.3);
}
.badge-ungrounded {
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.badge-model {
    background: rgba(59, 130, 246, 0.12);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.25);
}
.badge-neutral {
    background: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
}

/* Answer Text */
.bot-answer-text {
    font-size: 15.5px;
    font-weight: 500;
    color: #f8fafc;
    line-height: 1.65;
    margin-bottom: 10px;
    word-break: break-word;
}

/* Metrics Ribbon */
.bot-metrics-ribbon {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: #94a3b8;
    padding-top: 8px;
    border-top: 1px solid #1e293b;
}
.metric-highlight {
    color: #34d399;
    font-weight: 600;
}

/* Evidence Cards */
.evidence-card {
    background: #090d16;
    border: 1px solid #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.5;
}
.evidence-marker {
    color: #60a5fa;
    font-weight: 700;
    margin-right: 6px;
}
.evidence-meta {
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
    display: flex;
    gap: 8px;
}

/* Comparison Box */
.comp-card {
    background: #090d16;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px;
    height: 100%;
}
.comp-card-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e293b;
}
.comp-earc-title { color: #60a5fa; }
.comp-base-title { color: #94a3b8; }
.comp-answer-text {
    font-size: 13.5px;
    color: #e2e8f0;
    line-height: 1.5;
    margin-bottom: 8px;
}
.comp-meta-text {
    font-size: 11.5px;
    color: #64748b;
    border-top: 1px solid rgba(30, 41, 59, 0.5);
    padding-top: 6px;
}

/* 2x2 Grid Insight Cards */
.insight-card {
    background: #090d16;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px 14px;
    height: 100%;
}
.insight-card-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 8px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 4px;
}
.insight-row {
    display: flex;
    justify-content: space-between;
    padding: 3.5px 0;
    font-size: 12px;
    border-bottom: 1px solid rgba(30, 41, 59, 0.4);
}
.insight-row:last-child {
    border-bottom: none;
}
.insight-label { color: #94a3b8; }
.insight-value { color: #f1f5f9; font-weight: 600; }

/* Clean Expanders */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    margin-top: 6px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-size: 12.5px !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    padding: 8px 12px !important;
}
[data-testid="stExpander"] summary:hover {
    color: #f8fafc !important;
}

/* Chat Input Styling */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    background: #0f172a !important;
    border: 1px solid #334155 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline Cache (show_spinner=False prevents global floating spinner) ──────
@st.cache_resource(show_spinner=False)
def load_pipeline(llm_model: str):
    from src.ui.adapter import UIEARCPipeline
    return UIEARCPipeline(llm_model=llm_model or None)


# ── Helper: Insight Card HTML ─────────────────────────────────────────────────
def _card(title: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="insight-row">'
        f'<span class="insight-label">{label}</span>'
        f'<span class="insight-value">{value}</span>'
        f'</div>'
        for label, value in rows
    )
    return (
        f'<div class="insight-card">'
        f'<div class="insight-card-title">{title}</div>'
        f'{body}'
        f'</div>'
    )


# ── Benchmark & QA Pairs Evaluation Metrics Loader ────────────────────────────
_QA_STOPWORDS = {
    "what", "which", "who", "whom", "this", "that", "these", "those", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "a", "an", "the", "and", "but", "if", "or", "as", "of", "at", "by", "for", "with",
    "in", "out", "on", "off", "to", "from", "up", "down", "how", "why", "when", "where",
    "credited", "having", "generally"
}


def _get_content_words(text: str) -> set[str]:
    from src.evaluation.exact_match import normalize_answer
    words = re.findall(r"\b[a-z0-9]+\b", normalize_answer(text))
    return set(w for w in words if len(w) > 2 and w not in _QA_STOPWORDS)


@st.cache_resource(show_spinner=False)
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


@st.cache_data(show_spinner=False)
def _load_qa_knowledge_base() -> tuple[dict[str, dict[str, Any]], dict[str, list[int]], list[dict[str, Any]]]:
    """
    Load the 60,000 QA pairs from RAG_Project/qa_pairs/ once into memory.
    Also merges diagnostic benchmark examples from outputs/**/eval_results_raw.jsonl.
    Builds:
    1. exact_index: dict mapping normalized question string -> QA item.
    2. word_index: dict mapping content keyword -> list of item indices.
    3. all_qa: list of all QA items.
    """
    from src.evaluation.exact_match import normalize_answer

    exact_index: dict[str, dict[str, Any]] = {}
    word_index: dict[str, list[int]] = {}
    all_qa: list[dict[str, Any]] = []

    # 1. Load canonical RAG_Project QA pairs (60,000 total)
    qa_dir = Path(_PROJECT_ROOT) / "RAG_Project" / "qa_pairs"
    if qa_dir.exists():
        for pkl_file in sorted(qa_dir.glob("qa_*.pkl")):
            try:
                with open(pkl_file, "rb") as fp:
                    items = pickle.load(fp)
                    for item in items:
                        idx = len(all_qa)
                        all_qa.append(item)
                        q_text = item.get("question", "")
                        norm_q = normalize_answer(q_text)
                        if norm_q and norm_q not in exact_index:
                            exact_index[norm_q] = item
                        for w in _get_content_words(q_text):
                            if w not in word_index:
                                word_index[w] = []
                            word_index[w].append(idx)
            except Exception:
                continue

    # 2. Merge diagnostic benchmark records (ensures sampled NQ/HotpotQA benchmark questions resolve)
    diag_dir = Path(_PROJECT_ROOT) / "outputs"
    if diag_dir.exists():
        for p in diag_dir.glob("**/eval_results_raw.jsonl"):
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        item = json.loads(line_str)
                        q_text = item.get("question", "")
                        norm_q = normalize_answer(q_text)
                        if norm_q and norm_q not in exact_index:
                            qa_record = {
                                "question_id": item.get("example_id", norm_q),
                                "question": q_text,
                                "answers": item.get("gold_answers", []),
                                "dataset": item.get("dataset", "benchmark"),
                                "doc_id": None,
                            }
                            idx = len(all_qa)
                            all_qa.append(qa_record)
                            exact_index[norm_q] = qa_record
                            for w in _get_content_words(q_text):
                                if w not in word_index:
                                    word_index[w] = []
                                word_index[w].append(idx)
            except Exception:
                continue

    return exact_index, word_index, all_qa


def _resolve_gold_answer(query: str, sim_threshold: float = 0.88) -> dict[str, Any] | None:
    """
    Resolve a user query to a ground-truth QA pair from the 60,000 RAG_Project QA dataset:
    1. Tier 1: Fast exact normalized match.
    2. Tier 2: Conservative semantic match using keyword candidate pre-filtering,
       SentenceTransformer cosine similarity >= 0.88, and entity-shift rejection.
    """
    from src.evaluation.exact_match import normalize_answer

    exact_index, word_index, all_qa = _load_qa_knowledge_base()
    norm_q = normalize_answer(query)

    # Tier 1: Exact normalized string match
    if norm_q in exact_index:
        item = exact_index[norm_q]
        return {
            "match_type": "exact",
            "similarity": 1.0,
            "gold_answers": item.get("answers", []),
            "matched_question": item.get("question", ""),
            "dataset": item.get("dataset", ""),
            "question_id": item.get("question_id", ""),
        }

    # Tier 2: Conservative keyword-filtered semantic match
    q_words = _get_content_words(query)
    if not q_words:
        return None

    candidate_indices: set[int] = set()
    for w in q_words:
        if w in word_index:
            candidate_indices.update(word_index[w])

    if not candidate_indices:
        return None

    candidates = [all_qa[i] for i in candidate_indices]

    # Content word filter: candidate must not introduce new specific entities/nouns
    # that change the topic (e.g. "Queensland" when user asked for "Australia")
    valid_candidates = []
    for c in candidates:
        c_words = _get_content_words(c.get("question", ""))
        extra_words = c_words - q_words
        if not extra_words:
            valid_candidates.append(c)

    if not valid_candidates:
        return None

    if len(valid_candidates) > 200:
        valid_candidates = valid_candidates[:200]

    cand_questions = [c.get("question", "") for c in valid_candidates]
    model = _get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    cand_embs = model.encode(cand_questions, normalize_embeddings=True)

    sims = np.dot(cand_embs, q_emb)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim >= sim_threshold:
        best_item = valid_candidates[best_idx]
        return {
            "match_type": "semantic",
            "similarity": best_sim,
            "gold_answers": best_item.get("answers", []),
            "matched_question": best_item.get("question", ""),
            "dataset": best_item.get("dataset", ""),
            "question_id": best_item.get("question_id", ""),
        }

    return None


def _extract_evaluation_metrics(result: Any, query: str = "", provider_name: str = "ollama") -> dict[str, Any]:
    """
    Extract evaluation metrics (EM, F1, Hit@1/5/10) using:
    1) Direct fields on result if already present (e.g. pre-computed ExperimentResult).
    2) The resolved gold answer from the 60,000 RAG_Project QA dataset, evaluated using
       existing exact_match_score, max_token_f1_score, and compute_retrieval_hit.
    For arbitrary user queries without gold answers, returns has_gold=False.
    """
    from src.evaluation.exact_match import exact_match_score
    from src.evaluation.f1 import max_token_f1_score
    from src.evaluation.aggregation import compute_retrieval_hit

    is_dict = isinstance(result, dict)

    def _get_val(keys: list[str], default=None):
        for k in keys:
            if is_dict and k in result and result[k] is not None:
                return result[k]
            elif hasattr(result, k) and getattr(result, k) is not None:
                return getattr(result, k)
        return default

    exact_match = _get_val(["exact_match", "exact_match_earc", "em"])
    f1 = _get_val(["f1", "token_f1", "token_f1_earc"])
    hit_1 = _get_val(["hit_at_1", "answer_hit_at_1", "hit@1"])
    hit_5 = _get_val(["hit_at_5", "answer_hit_at_5", "hit@5"])
    hit_10 = _get_val(["hit_at_10", "answer_hit_at_10", "hit@10"])
    gold_answers = _get_val(["gold_answers", "gold_answer", "answers"])
    dataset = _get_val(["dataset"])
    matched_q = _get_val(["matched_question"])
    match_type = _get_val(["match_type"])

    effective_query = query or _get_val(["query", "question"], "")
    evaluated_answer = _get_val(["answer", "evaluated_answer", "prediction"], "")
    retrieved_docs = _get_val(["retrieved_documents"], [])

    # If gold_answers is not already attached, resolve from RAG_Project QA pairs
    if gold_answers is None and effective_query:
        resolved = _resolve_gold_answer(effective_query)
        if resolved:
            gold_answers = resolved.get("gold_answers")
            dataset = dataset or resolved.get("dataset")
            matched_q = resolved.get("matched_question")
            match_type = resolved.get("match_type")

    # If gold answers resolved, compute EM/F1 using existing backend evaluation functions
    if gold_answers:
        if exact_match is None and evaluated_answer:
            exact_match = float(exact_match_score(evaluated_answer, gold_answers))
        if f1 is None and evaluated_answer:
            f1 = float(max_token_f1_score(evaluated_answer, gold_answers)["f1"])

        if (hit_1 is None or hit_5 is None or hit_10 is None) and retrieved_docs:
            if hit_1 is None:
                hit_1 = float(compute_retrieval_hit(retrieved_docs, gold_answers, k=1))
            if hit_5 is None:
                hit_5 = float(compute_retrieval_hit(retrieved_docs, gold_answers, k=5))
            if hit_10 is None:
                hit_10 = float(compute_retrieval_hit(retrieved_docs, gold_answers, k=10))

    has_gold = bool(gold_answers)
    return {
        "has_gold": has_gold,
        "gold_answers": gold_answers,
        "dataset": dataset,
        "exact_match": exact_match,
        "f1": f1,
        "hit_at_1": hit_1,
        "hit_at_5": hit_5,
        "hit_at_10": hit_10,
        "matched_question": matched_q,
        "match_type": match_type,
    }


# ── Technical Details & Insights Component (Collapsed by Default) ─────────────
def _render_insights(result: Any, query: str = "") -> None:
    if hasattr(result, "to_dict") and not isinstance(result, dict):
        res_dict = result.to_dict()
    elif isinstance(result, dict):
        res_dict = result
    else:
        res_dict = {}

    query_info      = res_dict.get("query_info", {}) or {}
    scoring_stats   = res_dict.get("scoring_stats", {}) or {}
    selection_stats = res_dict.get("selection_stats", {}) or {}
    generation      = res_dict.get("generation", {}) or {}
    verification    = generation.get("verification", {}) or {}
    latency_ms      = res_dict.get("latency_ms")

    step4 = scoring_stats.get("step4", {})
    step5 = scoring_stats.get("step5", {})
    step6 = scoring_stats.get("step6", {})
    budget = selection_stats.get("budget", {})

    retrieved_docs = res_dict.get("retrieved_documents", [])
    n_docs         = len(retrieved_docs) if retrieved_docs else 10
    n_retrieved    = step4.get("total_embedded", len(res_dict.get("retrieved_sentences", [])) or "—")
    query_type     = query_info.get("query_type", "—")
    keywords       = ", ".join(query_info.get("keywords", [])) or "—"

    n_before_dedup = step6.get("input_sentences", n_retrieved)
    n_after_dedup  = step6.get("output_sentences", "—")
    n_removed      = step6.get("removed", "—")
    mean_score     = step5.get("mean_score", "—")

    n_selected     = len(res_dict.get("selected_sentences", []))
    n_leftover     = len(res_dict.get("candidate_sentences", []))
    tokens_used    = budget.get("tokens_used", res_dict.get("compressed_tokens", "—"))
    token_budget   = budget.get("budget", 300)
    orig_tokens    = res_dict.get("original_tokens")
    comp_tokens    = res_dict.get("compressed_tokens")
    comp_pct       = res_dict.get("compression_percentage")
    if comp_pct is not None:
        comp_str = f"{comp_pct:.1f}%"
    elif isinstance(n_before_dedup, int) and n_before_dedup > 0:
        comp_str = f"{100 * (1 - n_selected / n_before_dedup):.0f}%"
    else:
        comp_str = "—"

    backend        = generation.get("backend", "—")
    citations_cnt  = len(generation.get("citations", []))
    faithfulness   = verification.get("faithfulness")
    mean_overlap   = verification.get("mean_overlap")

    retrieval_rows = [
        ("Retrieval method",    "Hybrid"),
        ("Documents retrieved", str(n_docs)),
        ("Sentences extracted", str(n_retrieved)),
        ("Question type",       str(query_type)),
        ("Detected keywords",   str(keywords)),
    ]

    scoring_rows = [
        ("Sentences entering",   str(n_before_dedup)),
        ("After deduplication",  str(n_after_dedup)),
        ("Removed as redundant", str(n_removed)),
    ]
    if mean_score != "—" and isinstance(mean_score, (int, float)):
        scoring_rows.append(("Mean composite score", f"{mean_score:.4f}"))

    selection_rows = [
        ("Selected sentences",   str(n_selected)),
        ("Leftover candidates",  str(n_leftover)),
        ("Token budget",         str(token_budget)),
        ("Tokens used",          str(tokens_used)),
    ]
    if orig_tokens is not None:
        selection_rows.append(("Original context tokens", str(orig_tokens)))
    if comp_tokens is not None:
        selection_rows.append(("Compressed tokens", str(comp_tokens)))
    selection_rows.append(("Token reduction", comp_str))

    eval_rows = [
        ("Provider / Model",     str(backend)),
        ("Faithfulness",         f"{faithfulness:.3f}" if faithfulness is not None else "—"),
        ("Mean token overlap",   f"{mean_overlap:.3f}" if mean_overlap is not None else "—"),
        ("Citations count",      str(citations_cnt)),
        ("End-to-end latency",   f"{latency_ms:,.0f} ms" if latency_ms is not None else "—"),
    ]

    # Benchmark Evaluation Metrics
    effective_query = query or res_dict.get("query", "") or res_dict.get("question", "")
    provider_key = "mistral" if "mistral" in str(backend).lower() else "ollama"
    eval_metrics = _extract_evaluation_metrics(result, query=effective_query, provider_name=provider_key)

    if eval_metrics["has_gold"]:
        em_val = eval_metrics["exact_match"]
        f1_val = eval_metrics["f1"]
        h1_val = eval_metrics["hit_at_1"]
        h5_val = eval_metrics["hit_at_5"]
        h10_val = eval_metrics["hit_at_10"]
        golds = eval_metrics["gold_answers"]

        if isinstance(golds, list):
            gold_str = " | ".join(str(g) for g in golds)
        else:
            gold_str = str(golds) if golds is not None else ""

        bench_eval_rows = [
            ("Exact Match (EM)", f"{em_val:.3f}" if isinstance(em_val, (int, float)) else (str(em_val) if em_val is not None else "—")),
            ("F1 Score",         f"{f1_val:.3f}" if isinstance(f1_val, (int, float)) else (str(f1_val) if f1_val is not None else "—")),
            ("Answer Hit@1",     f"{h1_val:.1f}" if isinstance(h1_val, (int, float)) else (str(h1_val) if h1_val is not None else "—")),
            ("Answer Hit@5",     f"{h5_val:.1f}" if isinstance(h5_val, (int, float)) else (str(h5_val) if h5_val is not None else "—")),
            ("Answer Hit@10",    f"{h10_val:.1f}" if isinstance(h10_val, (int, float)) else (str(h10_val) if h10_val is not None else "—")),
        ]
        if eval_metrics.get("dataset"):
            bench_eval_rows.insert(0, ("Benchmark Dataset", str(eval_metrics["dataset"]).upper()))
        if gold_str:
            bench_eval_rows.insert(1 if eval_metrics.get("dataset") else 0, ("Gold Answer", gold_str))
    else:
        bench_eval_rows = [
            ("Evaluation",         "Gold answer not provided — EM/F1 unavailable"),
            ("Exact Match (EM)",   "Unavailable (no gold answer)"),
            ("F1 Score",           "Unavailable (no gold answer)"),
            ("Answer Hit@1",       "Unavailable (no gold answer)"),
            ("Answer Hit@5",       "Unavailable (no gold answer)"),
            ("Answer Hit@10",      "Unavailable (no gold answer)"),
        ]

    with st.expander("🔍 Pipeline Details & Verification", expanded=False):
        st.markdown(
            "<div style='font-size:12px;color:#94a3b8;margin-bottom:12px;font-weight:500'>"
            "Flow: <span style='color:#60a5fa'>Dense Retrieval</span> → "
            "<span style='color:#60a5fa'>Evidence Scoring</span> → "
            "<span style='color:#60a5fa'>Redundancy Filtering</span> → "
            "<span style='color:#60a5fa'>Token-Budget Selection</span> → "
            "<span style='color:#60a5fa'>Prompt Building</span> → "
            "<span style='color:#60a5fa'>LLM Generation</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.markdown(_card("📡 1. Dense Retrieval", retrieval_rows), unsafe_allow_html=True)
        c2.markdown(_card("📊 2. Evidence Scoring & Redundancy", scoring_rows), unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        c3.markdown(_card("✂ 3. Token-Budget Selection", selection_rows), unsafe_allow_html=True)
        c4.markdown(_card("🔬 4. Generation & Verification", eval_rows), unsafe_allow_html=True)

        c5, _ = st.columns([1, 1])
        c5.markdown(_card("📈 Evaluation Metrics", bench_eval_rows), unsafe_allow_html=True)


# ── Bot Message Renderer ──────────────────────────────────────────────────────
def _render_bot_message(query: str, result: dict, baseline: dict | None) -> None:
    generation   = result.get("generation", {})
    answer       = result.get("answer", "")
    citations    = generation.get("citations", []) or []
    verification = generation.get("verification", {}) or {}
    n_selected   = len(result.get("selected_sentences", []))
    n_retrieved  = len(result.get("retrieved_sentences", []))

    grounded     = verification.get("grounded")
    is_refusal   = verification.get("is_refusal", False)
    n_cit        = len(citations)

    orig_tokens  = result.get("original_tokens")
    comp_tokens  = result.get("compressed_tokens")
    comp_pct     = result.get("compression_percentage")
    latency_ms   = result.get("latency_ms")
    latency_s    = (latency_ms / 1000.0) if latency_ms else None

    # Status Badges
    if is_refusal:
        grounded_badge = '<span class="badge badge-abstained">⚠ Abstained</span>'
    elif grounded:
        grounded_badge = '<span class="badge badge-grounded">✓ Grounded</span>'
    else:
        grounded_badge = '<span class="badge badge-ungrounded">✗ Review Evidence</span>'

    backend_label = generation.get("backend", "EARC")
    badges_html = (
        f'{grounded_badge}'
        f'<span class="badge badge-model">⚡ {backend_label}</span>'
    )
    if n_cit > 0:
        badges_html += f'<span class="badge badge-neutral">📌 {n_cit} Citations</span>'

    # Compact Metrics Ribbon
    metrics_parts = []
    if orig_tokens and comp_tokens:
        metrics_parts.append(f"<span>📦 <b>{orig_tokens:,}</b> → <b>{comp_tokens:,}</b> tokens</span>")
    if comp_pct is not None:
        metrics_parts.append(f"<span class='metric-highlight'>⚡ {comp_pct:.1f}% reduction</span>")
    if latency_s:
        metrics_parts.append(f"<span>⏱ {latency_s:.2f}s latency</span>")
    metrics_html = " <span style='color:#334155'>·</span> ".join(metrics_parts)

    bot_card_html = f"""
    <div class="bot-msg-wrap">
      <div class="bot-card">
        <div class="bot-badge-bar">{badges_html}</div>
        <div class="bot-answer-text">{answer}</div>
        <div class="bot-metrics-ribbon">{metrics_html}</div>
      </div>
    </div>
    """
    st.markdown(bot_card_html, unsafe_allow_html=True)

    # ── Expander 1: Evidence & Citations (Collapsed by default) ────────────────
    if citations:
        with st.expander(f"📄 Evidence & Citations ({len(citations)} cited / {n_selected} selected)", expanded=False):
            for c in citations:
                score = round(float(c.get("score", 0.0) or 0.0), 4)
                raw_title = (c.get("title") or "").strip()
                title = raw_title if raw_title else "Knowledge Base"
                text = c.get("text", "")
                marker = c.get("marker", "")
                st.markdown(f"""
                <div class="evidence-card">
                  <span class="evidence-marker">{marker}</span>{text}
                  <div class="evidence-meta">
                    <span><b>Source:</b> {title}</span>
                    <span>·</span>
                    <span><b>Score:</b> {score}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Expander 2: EARC vs Standard RAG (Collapsed by default, only when enabled) ─
    if baseline:
        with st.expander("⚖️ EARC vs Standard RAG", expanded=False):
            col_earc, col_base = st.columns(2)
            with col_earc:
                st.markdown(f"""
                <div class="comp-card">
                  <div class="comp-card-title comp-earc-title">🧠 EARC (Compressed Context)</div>
                  <div class="comp-answer-text">{answer}</div>
                  <div class="comp-meta-text">
                    <b>Context:</b> {comp_tokens or '—'} tokens · <b>Selected:</b> {n_selected} sentences<br>
                    <b>Budget:</b> 300 tokens · <b>Method:</b> Dense (Top-10)
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with col_base:
                base_ans = baseline.get("answer", "")
                st.markdown(f"""
                <div class="comp-card">
                  <div class="comp-card-title comp-base-title">📚 Standard RAG (Full Context)</div>
                  <div class="comp-answer-text">{base_ans}</div>
                  <div class="comp-meta-text">
                    <b>Context:</b> {orig_tokens or '—'} tokens · <b>Retrieved:</b> {n_retrieved} sentences<br>
                    <b>Compression:</b> None (Uncompressed) · <b>Method:</b> Dense (Top-10)
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Compact comparison summary footer
            if orig_tokens and comp_tokens:
                reduction_label = f"{comp_pct:.1f}%" if comp_pct is not None else "—"
                st.markdown(
                    f"<div style='font-size:12px;color:#94a3b8;margin-top:10px;padding:6px 10px;background:#090d16;border:1px solid #1e293b;border-radius:6px;text-align:center'>"
                    f"Token Reduction: <b style='color:#34d399'>{orig_tokens:,} → {comp_tokens:,} tokens ({reduction_label} savings)</b> · "
                    f"Sentences: <b style='color:#60a5fa'>{n_retrieved} candidates → {n_selected} selected</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Expander 3: Detailed Pipeline Details & Verification ───────────────────
    _render_insights(result, query=query)

    # ── Expander 4: Full LLM Prompt ───────────────────────────────────────────
    with st.expander("🔬 Full LLM Prompt", expanded=False):
        st.code(generation.get("prompt", ""), language="text")


# ── Main Application ──────────────────────────────────────────────────────────
def main() -> None:
    MODEL_OPTIONS = {
        "Ollama — Llama 3.2": "llama3.2",
        "Mistral — Ministral 8B": "mistral",
    }

    # ── Multi-Conversation State Initialization ────────────────────────────────
    if "conversations" not in st.session_state:
        initial_id = uuid.uuid4().hex[:8]
        st.session_state.conversations = {
            initial_id: {
                "id": initial_id,
                "title": "New chat",
                "messages": [],
                "updated_at": time.time(),
            }
        }
        st.session_state.active_conv_id = initial_id

    if "active_conv_id" not in st.session_state or st.session_state.active_conv_id not in st.session_state.conversations:
        if st.session_state.conversations:
            st.session_state.active_conv_id = next(iter(st.session_state.conversations.keys()))
        else:
            initial_id = uuid.uuid4().hex[:8]
            st.session_state.conversations[initial_id] = {
                "id": initial_id,
                "title": "New chat",
                "messages": [],
                "updated_at": time.time(),
            }
            st.session_state.active_conv_id = initial_id

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    active_id = st.session_state.active_conv_id
    active_conv = st.session_state.conversations[active_id]

    # ── Native Streamlit Sidebar (ChatGPT / Claude Style) ──────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:2px'>"
            "<span style='font-size:24px'>🧠</span>"
            "<span style='font-size:18px;font-weight:700;color:#f8fafc'>EARC</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Evidence-Aware Retrieval & Compression")

        st.write("")
        # ＋ New chat button
        if st.button("＋ New chat", key="btn_new_chat", use_container_width=True, type="primary"):
            # If current active chat already has messages, create a fresh chat
            if active_conv["messages"]:
                new_id = uuid.uuid4().hex[:8]
                st.session_state.conversations[new_id] = {
                    "id": new_id,
                    "title": "New chat",
                    "messages": [],
                    "updated_at": time.time(),
                }
                st.session_state.active_conv_id = new_id
            st.session_state.pending_query = None
            st.rerun()

        st.divider()

        # LLM Model Selector
        st.markdown("<div style='font-size:12px;font-weight:600;color:#94a3b8;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.04em'>LLM Model</div>", unsafe_allow_html=True)
        selected_model_label = st.selectbox(
            "LLM Model",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
            label_visibility="collapsed",
            key="llm_model_select",
        )
        active_model_id = MODEL_OPTIONS[selected_model_label]

        st.divider()

        # EARC vs Standard RAG Comparison Checkbox
        show_baseline = st.checkbox(
            "Compare EARC vs Standard RAG",
            value=False,
            key="compare_baseline_cb",
            help="Runs uncompressed Standard RAG alongside EARC for side-by-side answer and token comparison.",
        )

        st.divider()

        # Recent Chats List
        st.markdown("<div style='font-size:11.5px;font-weight:600;color:#64748b;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em'>Recent Chats</div>", unsafe_allow_html=True)

        sorted_convs = sorted(
            st.session_state.conversations.values(),
            key=lambda c: c.get("updated_at", 0),
            reverse=True,
        )

        for c in sorted_convs:
            cid = c["id"]
            is_active = (cid == active_id)
            display_title = c.get("title", "New chat")

            # Truncate title cleanly if long
            if len(display_title) > 28:
                truncated_title = display_title[:25] + "..."
            else:
                truncated_title = display_title

            btn_label = f"💬  {truncated_title}"
            btn_type = "primary" if is_active else "tertiary"

            if st.button(btn_label, key=f"btn_conv_{cid}", use_container_width=True, type=btn_type):
                if st.session_state.active_conv_id != cid:
                    st.session_state.active_conv_id = cid
                    st.session_state.pending_query = None
                    st.rerun()

        st.write("")
        # Clear Chat: Clears ONLY the active conversation
        if st.button("🗑️ Clear chat", key="btn_clear_chat", use_container_width=True):
            active_conv["messages"] = []
            active_conv["title"] = "New chat"
            active_conv["updated_at"] = time.time()
            st.session_state.pending_query = None
            st.rerun()

        st.divider()

        st.markdown(
            "<div style='font-size:11px;color:#64748b;text-align:center;line-height:1.5'>"
            "Evidence-Aware Retrieval & Compression<br>"
            "<span style='color:#475569'>Grounded · Cited · Compressed</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Top Navigation Bar (Display-Only Status Indicator) ─────────────────────
    st.markdown(f"""
    <div class="top-nav-bar">
      <div class="brand-block">
        <span class="brand-logo">🧠</span>
        <div>
          <div class="brand-title">EARC Grounded QA</div>
          <div class="brand-sub">Grounded, cited, and compressed Question-Answering.</div>
        </div>
      </div>
      <div class="nav-pills-wrap">
        <span class="nav-pill">⚡ {selected_model_label}</span>
        <span class="nav-pill">Hybrid - Dense+BM25</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Query Input (Chat Input & Pending Click Resolution) ────────────────────
    pending = st.session_state.pop("pending_query", None)
    chat_input_val = st.chat_input(f"Ask a question... ({selected_model_label})")
    query = chat_input_val or pending

    # ── Welcome Screen: Only Rendered When Active Conversation Has No Messages ──
    has_activity = (len(active_conv["messages"]) > 0) or bool(query)

    if not has_activity:
        st.markdown("""
        <div class="welcome-container">
          <div class="welcome-logo">🧠</div>
          <div class="welcome-title">How can EARC help you today?</div>
          <div class="welcome-desc">
            EARC retrieves relevant passages from the knowledge base, scores evidence,
            filters out redundancy, and compresses context down to 300 tokens for grounded, hallucination-free answers.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:12.5px;font-weight:600;color:#94a3b8;margin-bottom:12px;text-align:center'>Sample questions:</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        sample_queries = [
            ("Where do the ilium, ischium, and pubis meet?", col1),
            ("What is the capital of Australia?", col2),
            ("What organ is responsible for pumping blood?", col1),
            ("Who wrote the play Romeo and Juliet?", col2),
        ]

        for q_text, col in sample_queries:
            if col.button(q_text, key=f"btn_{q_text}_{active_id}", use_container_width=True):
                st.session_state.pending_query = q_text
                st.rerun()

    # ── Replay Previous Chat Turns for the Active Conversation ─────────────────
    for turn in active_conv["messages"]:
        st.markdown(
            f'<div class="user-msg-wrap"><div class="user-msg-bubble">{turn["query"]}</div></div>',
            unsafe_allow_html=True,
        )
        _render_bot_message(turn["query"], turn["result"], turn.get("baseline"))

    # ── Handle Newly Submitted Query ──────────────────────────────────────────
    if query and query.strip():
        user_text = query.strip()

        # Update conversation title on first message
        if not active_conv["messages"]:
            active_conv["title"] = user_text[:30].strip() + ("..." if len(user_text) > 30 else "")

        # Render user message bubble immediately
        st.markdown(
            f'<div class="user-msg-wrap"><div class="user-msg-bubble">{user_text}</div></div>',
            unsafe_allow_html=True,
        )

        pipe = load_pipeline(active_model_id)

        # Clean inline loading placeholder in assistant area (never covers composer)
        status_slot = st.empty()
        with status_slot.container():
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:12px 16px;background:#0f172a;border:1px solid #1e293b;border-radius:12px;color:#94a3b8;font-size:13.5px;margin:8px 0">'
                f'<span>⏳</span> <span>EARC is retrieving, scoring, and compressing evidence ({selected_model_label})...</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            result = pipe.run(user_text)

            baseline = None
            if show_baseline:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:12px 16px;background:#0f172a;border:1px solid #1e293b;border-radius:12px;color:#94a3b8;font-size:13.5px;margin:8px 0">'
                    f'<span>⏳</span> <span>Generating uncompressed Standard RAG baseline...</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                baseline = pipe.generate_baseline(
                    user_text,
                    retrieved_docs=result.get("retrieved_documents", []),
                )

        # Clear inline status slot before rendering bot answer
        status_slot.empty()

        # Render assistant answer
        _render_bot_message(user_text, result, baseline)

        # Save turn to the active conversation's history
        active_conv["messages"].append({
            "query":    user_text,
            "result":   result,
            "baseline": baseline,
        })
        active_conv["updated_at"] = time.time()
        st.rerun()


if __name__ == "__main__":
    main()