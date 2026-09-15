import random
import logging
from typing import List, Dict, Any, Optional

from src.data.schemas import QAExample
from src.data.loaders import load_dataset, load_nq_passage_pool
from src.config.settings import DataConfig
from src.utils.io import ensure_dir, write_jsonl

logger = logging.getLogger(__name__)

def sample_dataset(examples: List[QAExample], n: int, seed: int) -> List[QAExample]:
    random.seed(seed)
    if n > len(examples):
        logger.warning(f"Requested {n} samples but dataset only has {len(examples)} examples.")
        return examples
    
    sampled = random.sample(examples, n)
    logger.info(f"Sampled {n} examples from dataset of size {len(examples)}")
    return sampled

def sample_with_replacement(examples: List[QAExample], n: int, seed: int) -> List[QAExample]:
    random.seed(seed)
    if not examples:
        raise ValueError("Cannot sample from an empty dataset.")
    
    sampled = random.choices(examples, k=n)
    logger.info(f"Sampled {n} examples with replacement from dataset of size {len(examples)}")
    return sampled

def validate_sample(examples: List[QAExample], expected_n: int, dataset_name: str) -> bool:
    count = len(examples)
    if count != expected_n:
        logger.error(f"Validation failed for {dataset_name}: expected {expected_n} examples, got {count}")
        return False
        
    ids = set()
    duplicates = 0
    missing_context = 0
    malformed = 0
    
    for ex in examples:
        if ex.id in ids:
            duplicates += 1
        ids.add(ex.id)
        
        if not ex.question or not ex.answers:
            malformed += 1
            
        if dataset_name.lower() not in ['nq', 'nq_open'] and not ex.documents:
            missing_context += 1
            
    logger.info(f"Sample validation for {dataset_name}: Count={count}, Duplicates={duplicates}, "
                f"Missing Context={missing_context}, Malformed={malformed}")
    
    if malformed > 0:
        return False
        
    return True

def prepare_all_datasets(
    config: DataConfig,
    seed: Optional[int] = None,
    n_examples: Optional[int] = None,
    output_dir: Optional[str] = None,
    strip_metadata: bool = True,
) -> Dict[str, List[QAExample]]:
    """
    Load, sample, validate, and persist each configured dataset.

    seed / n_examples / output_dir override whatever is in `config` when
    provided (this is what scripts/prepare_data.py's CLI flags feed in).
    strip_metadata=True (default) clears the free-form `metadata` dict on
    every example/document before writing to disk, so processed files stay
    lean (id/dataset/question/answers/documents only).
    """
    datasets_to_load = getattr(config, 'datasets', ['nq', 'hotpotqa', 'triviaqa'])
    result = {}

    seed = seed if seed is not None else getattr(config, 'seed', 42)
    n_samples = n_examples if n_examples is not None else getattr(config, 'n_examples_per_dataset', 5000)
    cache_dir = getattr(config, 'cache_dir', None)
    output_dir = output_dir or getattr(config, 'processed_dir', 'data/processed')
    split_map = {
        'nq': getattr(config, 'nq_split', 'train'),
        'hotpotqa': getattr(config, 'hotpotqa_split', 'train'),
        'triviaqa': getattr(config, 'triviaqa_split', 'train'),
    }

    ensure_dir(output_dir)

    for ds_name in datasets_to_load:
        logger.info(f"Preparing dataset: {ds_name}")
        try:
            examples = load_dataset(ds_name, split=split_map.get(ds_name, 'train'), cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to load dataset {ds_name}: {e}")
            raise

        valid_examples = [ex for ex in examples if ex.question and ex.answers]
        skipped = len(examples) - len(valid_examples)

        if not valid_examples:
            logger.error(f"No valid examples found for dataset {ds_name}")
            raise ValueError(f"No valid examples found for dataset {ds_name}")

        if len(valid_examples) < n_samples:
            logger.warning(f"Insufficient examples for {ds_name}. Needed {n_samples}, found {len(valid_examples)}. Using replacement.")
            sampled = sample_with_replacement(valid_examples, n_samples, seed)
        else:
            sampled = sample_dataset(valid_examples, n_samples, seed)

        is_valid = validate_sample(sampled, n_samples, ds_name)
        if not is_valid:
            logger.warning(f"Validation issues found for sample of {ds_name}")

        if strip_metadata:
            for ex in sampled:
                ex.metadata = {}
                for doc in ex.documents:
                    doc.metadata = {}

        logger.info(f"Prepared {ds_name}: Skipped={skipped} invalid examples, Selected={len(sampled)}.")
        result[ds_name] = sampled

        out_path = f"{output_dir.rstrip('/')}/{ds_name}.jsonl"
        write_jsonl(out_path, [ex.to_dict() for ex in sampled])
        logger.info(f"Wrote {len(sampled)} examples for {ds_name} to {out_path}")

        if ds_name == 'nq':
            # NQ Open has no context of its own -- pull a shared passage pool
            # (matches the old data_builder.py's nq_passages step) so NQ isn't
            # the only dataset with zero retrievable documents.
            n_passages = getattr(config, 'nq_passage_pool_size', n_samples)
            passages = load_nq_passage_pool(n_passages, cache_dir)
            passages_path = f"{output_dir.rstrip('/')}/nq_passages.jsonl"
            write_jsonl(passages_path, [p.to_dict() for p in passages])
            logger.info(f"Wrote {len(passages)} NQ passages to {passages_path}")

    return result
