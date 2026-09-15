import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    logger.info(f"Set random seed to {seed} across all backends.")
