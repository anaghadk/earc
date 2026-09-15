import sys
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.config.settings import EARCConfig, load_config

def test_default_config():
    config = EARCConfig()
    assert config.model.name is not None

def test_yaml_loading(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_content = """
model:
  name: "custom-model"
"""
    yaml_file.write_text(yaml_content)
    
    config = load_config(str(yaml_file))
    assert config.model.name == "custom-model"

def test_override():
    config = load_config(overrides={"model.name": "overridden-model"})
    assert config.model.name == "overridden-model"

def test_alpha_beta_sum_warning():
    with pytest.warns(UserWarning):
        config = EARCConfig()
        config.compression.alpha = 0.8
        config.compression.beta = 0.3
        config.model_post_init(None)
