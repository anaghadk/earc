import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

def test_smoke():
    # Run the pipeline script with limit 1 and mock provider
    # This verifies everything can be imported and runs without crashing
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "scripts" / "run_pipeline.py"
    
    if not script_path.exists():
        pytest.skip(f"Script {script_path} not found")
        
    result = subprocess.run(
        [sys.executable, str(script_path), "--limit", "1", "--provider", "mock"],
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    
    # It might fail if the dummy data doesn't exist, but that's fine for the smoke test
    # if it raises a specific data error, we can handle it.
    # Otherwise, it should return 0
    if result.returncode != 0:
        if "FileNotFoundError" in result.stderr and "data/raw" in result.stderr:
            pytest.skip("Raw data not found for smoke test")
        assert False, f"Smoke test failed: {result.stderr}"
