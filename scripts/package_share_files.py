"""
scripts/package_share_files.py

Packages all files requested in Part 2 into a single self-contained zip file:
  C:/earc/earc_share_deliverables.zip

Files included:
1. Four new HotpotQA repetition runs:
   - runs/hotpot_reps/earc_default__hotpotqa__llama__rep2.jsonl
   - runs/hotpot_reps/earc_default__hotpotqa__llama__rep3.jsonl
   - runs/hotpot_reps/earc_default__hotpotqa__ministral__rep2.jsonl
   - runs/hotpot_reps/earc_default__hotpotqa__ministral__rep3.jsonl
2. Corrected result tables:
   - results_strict_corrected.csv
   - paired_tests_corrected.csv
   - retention_attribution_corrected.csv
   - fresh_vs_stored_corrected.csv
3. All per-question run files (JSONL):
   - main runs (Standard RAG, Top-5, LLMLingua-2, EARC) from runs/baseline/
   - ablation A1-A4 from runs/exp1_ablation/
   - beta sweeps (B1-B3) from runs/exp1_ablation/
   - budget sweeps from runs/exp2_budget/
   - October reruns from runs/rerun_default/predictions/
4. hitk.csv (Hit@10 per question).
"""

import os
import sys
import glob
import zipfile
from pathlib import Path

PROJECT_ROOT = Path("C:/earc")
ZIP_OUT = PROJECT_ROOT / "earc_share_deliverables.zip"

def package_all():
    print(f"Building zip archive: {ZIP_OUT}...")
    file_manifest = []

    # 1. Four new HotpotQA rep files
    hotpot_reps = sorted((PROJECT_ROOT / "runs/hotpot_reps").glob("*.jsonl"))
    for f in hotpot_reps:
        file_manifest.append((f, f"runs/hotpot_reps/{f.name}"))

    # 2. Corrected tables
    tdir = PROJECT_ROOT / "runs/rerun_default/tables"
    for tname in [
        "results_strict_corrected.csv",
        "paired_tests_corrected.csv",
        "retention_attribution_corrected.csv",
        "fresh_vs_stored_corrected.csv"
    ]:
        fpath = tdir / tname
        if fpath.exists():
            file_manifest.append((fpath, f"tables/{tname}"))
        else:
            print(f"Warning: {fpath} not found!")

    # 3. All per-question run files (JSONL)
    # Main baseline runs
    for f in sorted((PROJECT_ROOT / "runs/baseline").glob("*.jsonl")):
        file_manifest.append((f, f"runs/baseline/{f.name}"))

    # Ablation A1-A4 and beta sweeps B1-B3
    for f in sorted((PROJECT_ROOT / "runs/exp1_ablation").glob("*.jsonl")):
        file_manifest.append((f, f"runs/exp1_ablation/{f.name}"))

    # Budget sweeps
    for f in sorted((PROJECT_ROOT / "runs/exp2_budget").glob("*.jsonl")):
        file_manifest.append((f, f"runs/exp2_budget/{f.name}"))

    # October reruns
    for f in sorted((PROJECT_ROOT / "runs/rerun_default/predictions").glob("*.jsonl")):
        file_manifest.append((f, f"runs/rerun_default/predictions/{f.name}"))

    # 4. hitk.csv
    hitk_path = PROJECT_ROOT / "task1_raw_files/hitk.csv"
    if hitk_path.exists():
        file_manifest.append((hitk_path, "hitk.csv"))

    print(f"Total files to package: {len(file_manifest)}")

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arc in file_manifest:
            zf.write(src, arcname=arc)
            print(f"  Added {arc} ({src.stat().st_size} bytes)")

    print(f"\nSuccessfully created {ZIP_OUT}!")
    print(f"Final archive size: {ZIP_OUT.stat().st_size:,} bytes")

if __name__ == "__main__":
    package_all()
