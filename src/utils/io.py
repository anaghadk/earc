import json
import os
import tempfile
import subprocess
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    """Write list of dicts to jsonl file."""
    ensure_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read list of dicts from jsonl file."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a single record to jsonl file."""
    ensure_dir(path)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def save_json(path: str, data: Any) -> None:
    """Save data to JSON file."""
    ensure_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def atomic_write(path: str, data: Any, is_json: bool = True) -> None:
    """Write data atomically to a file using tempfile rename."""
    ensure_dir(path)
    dir_name = os.path.dirname(path) or '.'
    fd, temp_path = tempfile.mkstemp(dir=dir_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            if is_json:
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                f.write(data)
        os.replace(temp_path, path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def save_config_snapshot(config: Any, output_dir: str) -> None:
    """Save configuration snapshot for reproducibility."""
    path = os.path.join(output_dir, 'config.json')
    ensure_dir(path)
    try:
        if hasattr(config, 'model_dump'):
            data = config.model_dump()
        elif hasattr(config, 'dict'):
            data = config.dict()
        else:
            data = vars(config)
        atomic_write(path, data)
    except Exception as e:
        logger.error(f"Failed to save config snapshot: {e}")

def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        return commit
    except Exception:
        return None
