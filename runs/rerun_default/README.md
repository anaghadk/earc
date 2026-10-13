# EARC Consolidated Rerun Deliverables (`runs/rerun_default/`)

This directory is the single, self-contained final deliverables repository for the **EARC Correction and Rerun Workflow**, following the authoritative correction guide (*“EARC additional experiments: corrections and reruns”*, Simon, Oct 10, 2026).

---

## 1. Directory Structure

```text
runs/rerun_default/
├── README.md                      # This consolidated deliverables guide
├── MANIFEST.json                  # Exhaustive catalog of all deliverables, provenance, and validation
├── CHECKLIST.md                   # Requirement-by-requirement audit matrix (COMPLETE / BLOCKED / INCOMPLETE)
├── commit.txt                     # Git commit snapshot, working-tree status, and hardware/software info
├── compare_rerun.txt              # Diagnostic output from compare_runs.py across all 12 fresh-vs-stored pairs
├── requirements_frozen.txt        # Frozen Python virtual environment dependencies (pip freeze)
│
├── predictions/                   # All 12 fresh 1,000-question generation runs (JSONL)
│   ├── earc_default_rerun__nq__llama3.2_3b.jsonl
│   ├── earc_default_rerun__hotpotqa__llama3.2_3b.jsonl
│   ├── earc_default_rerun__triviaqa__llama3.2_3b.jsonl
│   ├── topk5_rerun__nq__llama3.2_3b.jsonl
│   ├── topk5_rerun__hotpotqa__llama3.2_3b.jsonl
│   ├── topk5_rerun__triviaqa__llama3.2_3b.jsonl
│   ├── earc_default_rerun__nq__ministral_8b.jsonl
│   ├── earc_default_rerun__hotpotqa__ministral_8b.jsonl
│   ├── earc_default_rerun__triviaqa__ministral_8b.jsonl
│   ├── topk5_rerun__nq__ministral_8b.jsonl
│   ├── topk5_rerun__hotpotqa__ministral_8b.jsonl
│   └── topk5_rerun__triviaqa__ministral_8b.jsonl
│
├── tables/                        # Strictly rescored CSVs and statistical results
│   ├── fresh_runs_summary_strict.csv      # Strict EM/F1, tokens, and latency for the 12 fresh runs
│   ├── fresh_vs_stored_summary.csv        # Side-by-side fresh vs stored delta and context match rates
│   ├── results_strict.csv                 # All 90 recovered runs strictly rescored with SQuAD v1.1
│   ├── rescore_summary.csv                # High-level rescoring audit across all recovered experimental runs
│   ├── exp1_summary_strict.csv            # Experiment 1 architectural component ablations (A1-A5, B1-B3)
│   ├── exp2_summary_strict.csv            # Experiment 2 token budget scaling (b150-b600, topk_5)
│   ├── exp3_summary_strict.csv            # Experiment 3 LongLLMLingua baseline tracking
│   ├── retention_attribution_strict.csv   # Task 4 retention and 4-way error attribution (sums to 100%)
│   ├── comparisons.csv                    # Specification of paired comparison tests across 48 slots
│   ├── paired_tests_strict.csv            # Bootstrap 95% CIs, bootstrap p-values, and Wilcoxon p-values
│   └── compressor_settings.csv            # Task 6 comparison matrix of compressor parameters
│
├── figures/                       # Publication-ready plots in high-res PNG and vector PDF formats
│   ├── fig1_fresh_vs_stored_f1.{png,pdf}     # Stored vs fresh F1 scores across all 12 runs
│   ├── fig2_component_ablations.{png,pdf}    # Strict F1 across component ablations (A1 to A5)
│   ├── fig3_token_budget_curve.{png,pdf}     # HotpotQA token budget scaling curve (b150 - b600)
│   ├── fig4_error_attribution.{png,pdf}      # 4-way error breakdown (correct, miss, loss, gen error)
│   └── fig5_context_identity.{png,pdf}       # Byte-for-byte context identity match percentage
│
├── logs/                          # Verification and execution logs
│   ├── task3_llama_rerun.log              # Ollama Llama 3.2 3B 6-run execution log
│   ├── task3_ministral_rerun.log          # Mistral Client Ministral 8B 6-run execution log
│   ├── rescore_log.txt                    # Log of strict SQuAD v1.1 rescoring across all 90 files
│   ├── compare_rerun_log.txt              # Execution log of fresh-vs-stored comparisons
│   └── paired_tests_log.txt               # Output log of 10,000-resample bootstrap & Wilcoxon tests
│
└── reports/                       # Comprehensive investigative and technical reports
    ├── methodology_and_recovery.md        # Provenance, archive recovery, execution protocol
    ├── hotpotqa_discrepancy_investigation.md # Forensic analysis of HotpotQA fresh vs stored F1
    ├── missing_slots_audit.md             # Detailed status matrix of the 66 comparison slots
    └── implementation_questions_answered.md # Authoritative answers to the 10 implementation questions
```

---

## 2. Primary Files to Inspect First

1. **`reports/methodology_and_recovery.md`**: High-level overview of the recovery, rescoring, and fresh run methodology.
2. **`tables/fresh_runs_summary_strict.csv`**: Overview of strict EM and F1 performance across all 12 fresh runs.
3. **`tables/fresh_vs_stored_summary.csv`**: Direct side-by-side comparison between the lab PC stored runs and the fresh reruns.
4. **`reports/hotpotqa_discrepancy_investigation.md`**: Deep dive into the 99.70% context match and the reasoning margin decoding divergence on HotpotQA.
5. **`tables/paired_tests_strict.csv`**: Formal statistical significance tests (10,000 bootstrap resamples and Wilcoxon signed-rank tests) for all 48 completed paired comparison slots.
6. **`reports/missing_slots_audit.md`**: Forensic audit of why 48 slots are complete and 18 slots were unexecuted/blocked in the lab PC archive.
7. **`reports/implementation_questions_answered.md`**: Definitive architectural answers and code citations for the 10 implementation questions.
8. **`figures/`**: Visual verification of all experimental trends and distributions.

---

## 3. Strict Compliance Invariants

- **Original Archive Preservation**: `outputs.zip`, `task1_raw_files/`, and original source evidence remain 100% intact and unmutated.
- **Strict Scoring Invariant**: Strict EM is mathematically bounded by strict F1 ($EM \le F1$) across every evaluated JSONL line.
- **Zero Secrets Policy**: System verified with 0 exposed API keys across code, configs, logs, and Git tracking.
- **No Fabricated Data**: Missing lab PC runs are explicitly documented as `INCOMPLETE` / `BLOCKED` rather than approximated or fabricated.
