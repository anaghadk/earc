import argparse
import sys
from pathlib import Path
import logging
import os
import shutil
import importlib
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

def check_python():
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"PASS: Python version {version.major}.{version.minor}")
        return True
    print(f"FAIL: Python version {version.major}.{version.minor} (needs >=3.10)")
    return False

def check_package(pkg_name):
    try:
        importlib.import_module(pkg_name)
        print(f"PASS: {pkg_name} installed")
        return True
    except ImportError:
        print(f"FAIL: {pkg_name} not found")
        return False

def check_torch():
    try:
        import torch
        print(f"PASS: PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"PASS: CUDA available ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            print("PASS: MPS (Mac GPU) available")
        else:
            print("PASS: PyTorch running on CPU")
        return True
    except ImportError:
        print("FAIL: PyTorch not found")
        return False

def check_disk_space():
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    if free_gb > 10:
        print(f"PASS: {free_gb:.1f} GB free disk space")
        return True
    print(f"FAIL: Only {free_gb:.1f} GB free disk space (need 10+ GB)")
    return False

def check_ollama():
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags")
        if resp.status_code == 200:
            print("PASS: Ollama is running")
            return True
        print("FAIL: Ollama is running but returned non-200 status")
        return False
    except Exception:
        print("FAIL: Cannot connect to Ollama at localhost:11434")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify EARC environment")
    args = parser.parse_args()

    print("=== Environment Verification ===")
    
    checks = [
        check_python(),
        check_package("sentence_transformers"),
        check_package("transformers"),
        check_package("spacy"),
        check_package("faiss"),
        check_torch(),
        check_disk_space(),
        check_ollama()
    ]

    out_dir = Path("outputs")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        test_file = out_dir / ".test_write"
        test_file.touch()
        test_file.unlink()
        print("PASS: Output directory is writable")
        checks.append(True)
    except Exception as e:
        print(f"FAIL: Output directory not writable ({e})")
        checks.append(False)

    print("================================")
    if all(checks):
        print("ALL CRITICAL CHECKS PASSED")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
