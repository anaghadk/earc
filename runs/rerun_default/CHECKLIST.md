# EARC Rerun and Correction Workflow Checklist

This checklist maps every requirement from the authoritative correction guide (*“EARC additional experiments: corrections and reruns”*, Simon, Oct 10, 2026) to its final execution status (`COMPLETE`, `BLOCKED`, or `INCOMPLETE`) and exact deliverable file path in `runs/rerun_default/`.

---

## Master Requirements Checklist

| Requirement ID | Description | Guide Section | Status | Exact Deliverable Output Path | Notes & Invariants |
|---|---|:---:|:---:|---|---|
| **REQ-01** | Inventory and archive raw experimental artefacts | Task 1 | **COMPLETE** | `task1_raw_files.zip`<br>`task1_raw_files/inventory.txt` | 545 raw files indexed; original archive unmodified. |
| **REQ-02** | Strict SQuAD v1.1 rescoring of all 90 recovered runs | Task 2 | **COMPLETE** | `tables/results_strict.csv`<br>`tables/rescore_summary.csv`<br>`logs/rescore_log.txt` | Standard SQuAD v1.1 exact match & token F1. Verified $EM \le F1$ across all 90,000 predictions. |
| **REQ-03** | Experiment 1 Component Ablation Summary Table | Task 2 | **COMPLETE** | `tables/exp1_summary_strict.csv` | Covers A1-A5 and B1-B3 across all 6 pairs (48 runs). |
| **REQ-04** | Experiment 2 Token Budget Scaling Summary Table | Task 2 | **COMPLETE** | `tables/exp2_summary_strict.csv` | Covers b150, b300, b450, b600, and Top-k $k=5$ across all 6 pairs. |
| **REQ-05** | Experiment 3 LongLLMLingua Baseline Summary Table | Task 2 | **COMPLETE** | `tables/exp3_summary_strict.csv` | Tracked in archive (0 runs generated on lab PC). |
| **REQ-06** | Fresh rerun: NQ + Llama 3.2 3B (EARC default) | Task 3 | **COMPLETE** | `predictions/earc_default_rerun__nq__llama3.2_3b.jsonl` | 1,000 valid records; context match: 98.70%; $\Delta F1 = +0.57$. |
| **REQ-07** | Fresh rerun: HotpotQA + Llama 3.2 3B (EARC default) | Task 3 | **COMPLETE** | `predictions/earc_default_rerun__hotpotqa__llama3.2_3b.jsonl` | 1,000 valid records; context match: 99.70%; $\Delta F1 = -4.37$. |
| **REQ-08** | Fresh rerun: TriviaQA + Llama 3.2 3B (EARC default) | Task 3 | **COMPLETE** | `predictions/earc_default_rerun__triviaqa__llama3.2_3b.jsonl` | 1,000 valid records; context match: 98.20%; $\Delta F1 = +0.25$. |
| **REQ-09** | Fresh rerun: NQ + Llama 3.2 3B (Top-k k=5) | Task 3 | **COMPLETE** | `predictions/topk5_rerun__nq__llama3.2_3b.jsonl` | 1,000 valid records; context match: 100.00%; $\Delta F1 = -0.43$. |
| **REQ-10** | Fresh rerun: HotpotQA + Llama 3.2 3B (Top-k k=5) | Task 3 | **COMPLETE** | `predictions/topk5_rerun__hotpotqa__llama3.2_3b.jsonl` | 1,000 valid records; context match: 100.00%; $\Delta F1 = +0.63$. |
| **REQ-11** | Fresh rerun: TriviaQA + Llama 3.2 3B (Top-k k=5) | Task 3 | **COMPLETE** | `predictions/topk5_rerun__triviaqa__llama3.2_3b.jsonl` | 1,000 valid records; context match: 100.00%; $\Delta F1 = -2.02$. |
| **REQ-12** | Fresh rerun: NQ + Ministral 8B (EARC default) | Task 3 | **COMPLETE** | `predictions/earc_default_rerun__nq__ministral_8b.jsonl` | 1,000 valid records; context match: 98.70%; $\Delta F1 = +0.02$. |
| **REQ-13** | Fresh rerun: HotpotQA + Ministral 8B (EARC default) | Task 3 | **COMPLETE** | `predictions/earc_default_rerun__hotpotqa__ministral_8b.jsonl` | 1,000 valid records; context match: 99.70%; $\Delta F1 = -3.32$. |
| **REQ-14** | Fresh rerun: TriviaQA + Ministral 8B (EARC default) | Task 3 | **COMPLETE** | `predictions/earc_default_rerun__triviaqa__ministral_8b.jsonl` | 1,000 valid records; context match: 98.20%; $\Delta F1 = +0.62$. |
| **REQ-15** | Fresh rerun: NQ + Ministral 8B (Top-k k=5) | Task 3 | **COMPLETE** | `predictions/topk5_rerun__nq__ministral_8b.jsonl` | 1,000 valid records; context match: 100.00%; $\Delta F1 = +0.56$. |
| **REQ-16** | Fresh rerun: HotpotQA + Ministral 8B (Top-k k=5) | Task 3 | **COMPLETE** | `predictions/topk5_rerun__hotpotqa__ministral_8b.jsonl` | 1,000 valid records; context match: 100.00%; $\Delta F1 = +0.66$. |
| **REQ-17** | Fresh rerun: TriviaQA + Ministral 8B (Top-k k=5) | Task 3 | **COMPLETE** | `predictions/topk5_rerun__triviaqa__ministral_8b.jsonl` | 1,000 valid records; context match: 100.00%; $\Delta F1 = +0.32$. |
| **REQ-18** | Side-by-side fresh vs stored comparison analysis | Task 3 | **COMPLETE** | `compare_rerun.txt`<br>`tables/fresh_vs_stored_summary.csv`<br>`logs/compare_rerun_log.txt` | Evaluates all 12 pairs. Context identity: 98.2% to 100.0%. |
| **REQ-19** | HotpotQA discrepancy investigation report | Section 4 | **COMPLETE** | `reports/hotpotqa_discrepancy_investigation.md` | Proves 99.70% context match; identifies tight-budget multi-hop decoding sensitivity. |
| **REQ-20** | Recomputed retention and error attribution | Task 4 | **COMPLETE** | `tables/retention_attribution_strict.csv` | 91 rows (90 runs + header). All rows satisfy Correct + Miss + Loss + GenError = 100.0%. |
| **REQ-21** | Paired statistical tests across completed comparisons | Task 5 | **COMPLETE** | `tables/comparisons.csv`<br>`tables/paired_tests_strict.csv`<br>`logs/paired_tests_log.txt` | 48 slots evaluated with 10,000 bootstrap resamples and Wilcoxon signed-rank tests. |
| **REQ-22** | Matched-budget EARC vs Top-k comparisons (6 slots) | Task 5 | **INCOMPLETE** | `reports/missing_slots_audit.md` | Context files exist in raw archive, but prediction JSONL files were never executed on lab PC. |
| **REQ-23** | Top-k k=2 comparisons (6 slots) | Task 5 | **INCOMPLETE** | `reports/missing_slots_audit.md` | Context files exist in raw archive, but prediction JSONL files were never executed on lab PC. |
| **REQ-24** | LongLLMLingua comparisons (6 slots) | Task 5 | **BLOCKED** | `reports/missing_slots_audit.md` | Completely absent from outputs.zip; requires external 5.6 GB download and KV monkey-patch. |
| **REQ-25** | Compressor settings comparison table | Task 6 | **COMPLETE** | `tables/compressor_settings.csv` | Full parameters for LLMLingua-2 vs LongLLMLingua with source citations. |
| **REQ-26** | Ten open implementation questions answered | Task 7 | **COMPLETE** | `reports/implementation_questions_answered.md` | Authoritative answers with exact file paths and code lines. |
| **REQ-27** | Frozen environment & reproduction snapshot | Task 6 & 7 | **COMPLETE** | `requirements_frozen.txt`<br>`commit.txt` | Python 3.13, pip dependencies, git hash, diff stat, OS and hardware specs. |
| **REQ-28** | Publication figures generation (PNG + PDF) | Section 6 | **COMPLETE** | `figures/fig1_fresh_vs_stored_f1.{png,pdf}`<br>`figures/fig2_component_ablations.{png,pdf}`<br>`figures/fig3_token_budget_curve.{png,pdf}`<br>`figures/fig4_error_attribution.{png,pdf}`<br>`figures/fig5_context_identity.{png,pdf}` | 5 publication-ready figures produced in both high-res PNG and vector PDF formats. |
| **REQ-29** | Final deliverables manifest | Section 1 | **COMPLETE** | `MANIFEST.json` | JSON catalog detailing purpose, provenance, status, and validation outcome for every file. |
