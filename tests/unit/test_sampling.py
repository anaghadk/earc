import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.utils.seed import set_seed
import random
import numpy as np

def test_deterministic():
    set_seed(42)
    val1 = random.random()
    np_val1 = np.random.rand()
    
    set_seed(42)
    val2 = random.random()
    np_val2 = np.random.rand()
    
    assert val1 == val2
    assert np_val1 == np_val2

def test_correct_count():
    items = list(range(10))
    set_seed(42)
    sampled = random.sample(items, 5)
    assert len(sampled) == 5

def test_insufficient_examples():
    items = list(range(2))
    with pytest.raises(ValueError):
        random.sample(items, 5)
