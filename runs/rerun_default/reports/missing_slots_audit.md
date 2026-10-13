# Audit of Comparison Slots and Missing Experiments

## 1. Overview and Matrix Definition

Per Task 5 of the authoritative correction guide, a comprehensive statistical evaluation requires testing 11 specific paired comparison types across all 6 dataset-generator combinations:
$$\text{Total Slots} = 11 \text{ comparison types} \times 6 \text{ dataset-generator pairs} = 66 \text{ comparison slots}$$

The 6 dataset-generator pairs are:
1. Natural Questions (NQ) + Llama 3.2 3B
2. Natural Questions (NQ) + Ministral 8B
3. HotpotQA + Llama 3.2 3B
4. HotpotQA + Ministral 8B
5. TriviaQA + Llama 3.2 3B
6. TriviaQA + Ministral 8B

---

## 2. Comprehensive Status Matrix (66 Slots)

| Slot ID Range | Comparison Label | Condition A | Condition B | Hypothesized Effect | Completed Slots | Blocked / Missing Slots | Status |
|---|---|---|---|---|:---:|:---:|:---:|
| 1 – 6 | `full_vs_sim_only` | A5 (Full EARC) | A1 (Sim only) | Contribution of evidence scoring + redundancy | 6 / 6 | 0 | **COMPLETE** |
| 7 – 12 | `full_vs_evidence_only` | A5 (Full EARC) | A2 (Evid only) | Contribution of semantic similarity | 6 / 6 | 0 | **COMPLETE** |
| 13 – 18 | `full_vs_no_redundancy` | A5 (Full EARC) | A3 (No filter) | Contribution of redundancy filter ($\tau=0.85$) | 6 / 6 | 0 | **COMPLETE** |
| 19 – 24 | `full_vs_sim_redundancy` | A5 (Full EARC) | A4 ($\beta=0$) | Contribution of evidence term with filter active | 6 / 6 | 0 | **COMPLETE** |
| 25 – 30 | `redundancy_alone` | A4 ($\beta=0$) | A1 (Sim only) | Redundancy filter isolated effect | 6 / 6 | 0 | **COMPLETE** |
| 31 – 36 | `evidence_alone` | A3 (No filter) | A1 (Sim only) | Evidence term isolated effect | 6 / 6 | 0 | **COMPLETE** |
| 37 – 42 | `earc300_vs_top5` | EARC $T=300$ | Top-k $k=5$ | Main paper uncompressed baseline comparison | 6 / 6 | 0 | **COMPLETE** |
| 43 – 48 | `earc_vs_llmlingua2` | EARC $T=300$ | LLMLingua-2 | Task-aware compression vs token classification | 6 / 6 | 0 | **COMPLETE** |
| 49 – 54 | `earc_matched_vs_top5` | EARC matched budget | Top-k $k=5$ | Matched-budget comparison (Table 3) | 0 / 6 | 6 | **INCOMPLETE** |
| 55 – 60 | `earc300_vs_top2` | EARC $T=300$ | Top-k $k=2$ | Top-k at equivalent token budget (~300 tok) | 0 / 6 | 6 | **INCOMPLETE** |
| 61 – 66 | `earc_vs_longllmlingua` | EARC $T=300$ | LongLLMLingua | Question-aware perplexity prompt compression | 0 / 6 | 6 | **BLOCKED** |
| **Total** | | | | | **48 / 66** | **18 / 66** | **48 COMPLETE, 18 INCOMPLETE/BLOCKED** |

All 48 completed slots were computed with 10,000 bootstrap resamples and Wilcoxon signed-rank tests, recorded in `runs/rerun_default/tables/paired_tests_strict.csv`.

---

## 3. Detailed Investigation of the 18 Missing Slots

### 3.1 Six Matched-Budget EARC vs Top-k k=5 Comparisons (`earc_matched_vs_top5`)
- **Required Comparison**: EARC compressed contexts matched to the exact average token budget of Top-k $k=5$:
  - NQ: Token budget $T = 727$ tokens
  - HotpotQA: Token budget $T = 598$ tokens
  - TriviaQA: Token budget $T = 843$ tokens
- **Physical Forensic Evidence**:
  - Context files exist in `task1_raw_files/contexts/`:
    - `ctx_earc_nq_b727.json`
    - `ctx_earc_hotpotqa_b598.json`
    - `ctx_earc_triviaqa_b843.json`
  - However, no prediction JSONL files for these matched budgets were ever executed or archived on the lab PC. The lab PC only evaluated budgets $b \in \{150, 300, 450, 600\}$.
- **Status**: **INCOMPLETE / ABSENT FROM ARCHIVE**.
- **Resolution**: Under Section 5 of the correction guide (*"Do not fabricate predictions, approximate missing results, or substitute incompatible configurations"*), this condition is accurately documented as missing from the lab archive rather than fabricated.

### 3.2 Six Top-k k=2 Comparisons (`earc300_vs_top2`)
- **Required Comparison**: EARC $T=300$ vs Top-k $k=2$ passages (which represent ~250–320 tokens, offering a passage-level matched budget baseline).
- **Physical Forensic Evidence**:
  - Context files exist in `task1_raw_files/contexts/`:
    - `ctx_topk_nq_k2.json`
    - `ctx_topk_hotpotqa_k2.json`
    - `ctx_topk_triviaqa_k2.json`
  - However, the lab PC only ran generators on $k=5$ (`pred_baseline_*_topk_*.strict.jsonl` and `pred_topk_*_k5_*.strict.jsonl`). No generator predictions were created for $k=2$.
- **Status**: **INCOMPLETE / ABSENT FROM ARCHIVE**.
- **Resolution**: Documented as unexecuted on the lab PC.

### 3.3 Six LongLLMLingua Comparisons (`earc_vs_longllmlingua`)
- **Required Comparison**: EARC $T=300$ vs LongLLMLingua baseline across all 6 pairs.
- **Physical Forensic Evidence**:
  - Completely absent from `outputs.zip`.
  - The script `scripts/run_journal_exp3_longllmlingua.py` specifies `microsoft/phi-2` (2.7B parameters) with monkey-patched KV cache (`llmlingua/models/phi2.py`).
  - Running this locally requires downloading a 5.6 GB Hugging Face model weight repository and allocating substantial VRAM/RAM with specific CUDA kernels not present in the standard CPU/PyTorch Windows environment.
- **Status**: **BLOCKED / ABSENT FROM ARCHIVE**.
- **Resolution**: Documented as blocked due to missing lab PC runs and heavyweight external dependency requirements.
