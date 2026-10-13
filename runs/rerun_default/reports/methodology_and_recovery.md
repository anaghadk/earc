# EARC Methodology and Recovery Report

## 1. Overview and Executive Summary

This report documents the experimental methodology, archival recovery, and validation workflow for the EARC (*Evidence-Aware Retrieval-Guided Prompt Compression*) framework, following the authoritative correction guide (*"EARC additional experiments: corrections and reruns"*, Simon, Oct 10, 2026).

The objective was to audit, recover, rescore, validate, and complete the full benchmark across three question answering datasets (**Natural Questions**, **HotpotQA**, and **TriviaQA**) using two generator backends (**Llama 3.2 3B** via Ollama and **Ministral 8B** via the Mistral client).

---

## 2. Archival Recovery and Provenance (Task 1)

All original experimental artefacts were recovered from the lab PC archive `outputs.zip` located in the repository root without modifying, overwriting, or mutating any existing files.

### Recovered Artefact Inventory
- **Archive Size**: 81,343,641 bytes (`outputs.zip`).
- **Inventory File**: Consolidated inventory generated in `task1_raw_files/inventory.txt` containing 545 entries with SHA/timestamps, file sizes, and paths.
- **Recovered Prediction Runs**: 90 complete `.jsonl` files (1,000 questions each, total 90,000 question predictions).
  - 24 baseline runs (`runs/baseline/`): Standard RAG, Top-k ($k=5$), LLMLingua-2, and EARC default ($b=300$).
  - 48 component ablation runs (`runs/exp1_ablation/`): Configurations A1 through A5 and B1 through B3.
  - 18 token budget scaling runs (`runs/exp2_budget/`): Budgets $b \in \{150, 300, 450, 600\}$.
- **Recovered Context Files**: 66 `.json` files in `task1_raw_files/contexts/` containing exact compressed sentence sequences and token allocations.
- **Retrieval Cache**: `retrieved_passages.pkl` in `task1_raw_files/retrieval_cache/`, providing pre-retrieved top-$K$ passage text and chunk IDs for all 3,000 evaluation questions.

---

## 3. Strict Rescoring Methodology (Task 2)

### EM and F1 Metrics Implementation
The original lab evaluation had used substring-based matching for Exact Match (EM), allowing lenient predictions where any mention of the gold answer triggered $EM = 1.0$, which produced anomalies where $EM > F1$. 

In accordance with Task 2:
1. `scripts/rescore_strict.py` was deployed unchanged to evaluate standard SQuAD v1.1 metrics (Rajpurkar et al., 2016).
2. **Text Normalization**:
   - Lowercased text.
   - Punctuation stripped (`string.punctuation`).
   - Articles removed (`\b(a|an|the)\b`).
   - Whitespace collapsed to single spaces.
3. **Exact Match (EM)**: Returns `1.0` if and only if normalized prediction equals normalized gold answer:
   $$\text{EM} = \max_{g \in \text{golds}} \mathbb{I}(\text{norm}(p) == \text{norm}(g))$$
4. **Token F1**: Precision, Recall, and harmonic mean computed over multiset token overlap:
   $$F_1 = \max_{g \in \text{golds}} \frac{2 \cdot |p_{\text{tokens}} \cap g_{\text{tokens}}|}{|p_{\text{tokens}}| + |g_{\text{tokens}}|}$$
5. **Strict Metric Invariant**: Strict EM strictly upper-bounded by strict F1 ($EM \le F1$) was verified for all 90 recovered runs and all 12 fresh runs (0 violations).

Rescoring outputs were consolidated into:
- `runs/rerun_default/tables/results_strict.csv`
- `runs/rerun_default/tables/rescore_summary.csv`
- `runs/rerun_default/tables/exp1_summary_strict.csv`
- `runs/rerun_default/tables/exp2_summary_strict.csv`
- `runs/rerun_default/tables/exp3_summary_strict.csv`

---

## 4. Fresh Rerun Protocol and Execution (Task 3)

### Scope of Fresh Reruns
Twelve fresh end-to-end runs of 1,000 questions each were executed under the exact prescribed protocol:
- **Configurations**:
  - EARC Default: $\alpha = 0.7$, $\beta = 0.3$, redundancy threshold $\tau = 0.85$, token budget $T = 300$.
  - Top-k Baseline: Top-5 retrieved passages without compression ($k = 5$).
- **Datasets**: Natural Questions (NQ), HotpotQA, TriviaQA.
- **Generators**:
  - Llama 3.2 3B (local Ollama instance, `temperature=0.0`, `seed=42`).
  - Ministral 8B (Mistral client, `temperature=0.0`, `seed=42`).

### Execution Hardware and Environment
- **Host System**: Windows 11 Enterprise (Build 26100), AMD Ryzen 7 7730U with Radeon Graphics, 16 GB RAM.
- **Python**: 3.13.0 (Windows virtual environment at `C:\earc\.venv`).
- **Ollama**: Serving `llama3.2:3b` locally on port 11434.
- **Mistral SDK**: Client communicating via authenticated HTTPS with dynamic key loading.
- **Zero Secrets Policy**: System verified with 0 exposed API keys across code, configs, logs, and Git tracking.

### Validation Outcomes
All 12 fresh JSONL files in `runs/rerun_default/predictions/` and `runs/rerun_default/` were validated:
- Exactly 1,000 lines per file.
- Exactly 1,000 unique `question_id` values per file matching canonical question sets.
- 0 null or empty prediction fields.
- Both `context` and `retrieved_ids` populated for 100% of rows.
- Strict $EM \le F1$ validated across all 12,000 predictions.

---

## 5. Summary of Experimental Findings

1. **Context Determinism**:
   - Top-k $k=5$ context identity between stored and fresh runs is **100.00%** on all datasets.
   - EARC default context identity is **98.20% to 99.70%**, proving algorithm fidelity and deterministic sentence selection across operating systems.
2. **Predictive Performance**:
   - On NQ and TriviaQA, fresh runs reproduce stored F1 within $\pm 0.6$ points across both generators.
   - On HotpotQA, multi-hop reasoning margins under tight 300-token compression produce decoding divergence (analyzed in detail in `hotpotqa_discrepancy_investigation.md`).
3. **Statistical Invariance**:
   - Paired bootstrap tests (10,000 resamples) and Wilcoxon signed-rank tests confirm that EARC significantly outperforms similarity-only ($A1$) and evidence-only ($A2$) ablations ($p < 0.001$).
