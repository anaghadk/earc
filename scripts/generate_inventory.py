"""
Generate inventory.txt exactly matching Task 1 specification:
git log -1 --format='%H %ci' > inventory.txt
git status --short >> inventory.txt
find . -name '*.jsonl' -size +0 -printf '%s %p\n' | sort -k2 >> inventory.txt
find . -name '*.csv' -printf '%s %p\n' | sort -k2 >> inventory.txt
find . -name '*.yaml' -o -name '*.yml' | sort >> inventory.txt
"""

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("C:/earc")
os.chdir(PROJECT_ROOT)

lines = []

# 1. git log -1 --format='%H %ci'
git_log = subprocess.check_output(["git", "log", "-1", "--format=%H %ci"], text=True).strip()
lines.append(git_log)

# 2. git status --short
git_status = subprocess.check_output(["git", "status", "--short"], text=True)
if git_status:
    lines.extend([l for l in git_status.splitlines() if l.strip()])

# 3. find . -name '*.jsonl' -size +0 -printf '%s %p\n' | sort -k2
jsonl_files = []
for root, _, files in os.walk("."):
    for f in files:
        if f.endswith(".jsonl"):
            p = os.path.join(root, f).replace("\\", "/")
            sz = os.path.getsize(p)
            if sz > 0:
                jsonl_files.append((sz, p))
jsonl_files.sort(key=lambda x: x[1])
for sz, p in jsonl_files:
    lines.append(f"{sz} {p}")

# 4. find . -name '*.csv' -printf '%s %p\n' | sort -k2
csv_files = []
for root, _, files in os.walk("."):
    for f in files:
        if f.endswith(".csv"):
            p = os.path.join(root, f).replace("\\", "/")
            sz = os.path.getsize(p)
            csv_files.append((sz, p))
csv_files.sort(key=lambda x: x[1])
for sz, p in csv_files:
    lines.append(f"{sz} {p}")

# 5. find . -name '*.yaml' -o -name '*.yml' | sort
yaml_files = []
for root, _, files in os.walk("."):
    for f in files:
        if f.endswith(".yaml") or f.endswith(".yml"):
            p = os.path.join(root, f).replace("\\", "/")
            yaml_files.append(p)
yaml_files.sort()
for p in yaml_files:
    lines.append(p)

inv_path = PROJECT_ROOT / "inventory.txt"
with open(inv_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# Copy to task1_raw_files
import shutil
shutil.copy2(inv_path, PROJECT_ROOT / "task1_raw_files/inventory.txt")

print(f"Generated {inv_path} with {len(lines)} lines")
