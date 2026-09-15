# Evidence-Aware Retrieval-Guided Prompt Compression (EARC)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)

## 1. What This Project Implements
Full implementation of the EARC paper: sentence-level evidence-aware prompt compression for RAG that scores candidates on semantic relevance + evidential weight, filters cross-document redundancy, and greedily packs into a token budget.

## 2. Relationship to the Paper
Reproduction of 'Evidence-Aware Retrieval-Guided Prompt Compression for Token-Efficient Retrieval-Augmented Generation'. The compression algorithm is preserved exactly.

## 3. Intentional Changes from Paper
- 300 -> 5,000 examples per dataset
- HotpotQA + TriviaQA -> NQ + HotpotQA + TriviaQA (15,000 total)
- Flan-T5-base generation -> Ollama (llama3.2) + Mistral API
- Same compression algorithm, different scale and generation backends

## 4. Dataset Sources
- NQ: nq_open (train split) from HuggingFace
- HotpotQA: hotpot_qa distractor (train split) from HuggingFace
- TriviaQA: trivia_qa rc (train split) from HuggingFace

## 5. Sampling Policy
5,000 per dataset, seed=42, deterministic selection, replacement strategy if examples are invalid

## 6. Environment Setup
Python 3.10+, pip install requirements, spacy download en_core_web_sm
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 7. CUDA Installation
CUDA 12.1+, PyTorch with CUDA support, nvidia-smi verification
```bash
nvidia-smi
```

## 8. Model Downloads
all-MiniLM-L6-v2 (auto-downloaded by sentence-transformers)

## 9. Ollama Setup
Install Ollama, `ollama pull llama3.2`, verify with `ollama list`
```bash
ollama pull llama3.2
ollama list
```

## 10. Mistral API Setup
Get API key from console.mistral.ai, set in .env file

## 11. Download Data
```bash
python scripts/download_data.py
```

## 12. Prepare Data
```bash
python scripts/prepare_data.py
```

## 13. Build Index
```bash
python scripts/build_index.py
```

## 14. Run Pipeline
```bash
python scripts/run_pipeline.py --dataset all --provider ollama
```

## 15. Run Baselines
```bash
python scripts/run_baselines.py --dataset all --provider ollama
```

## 16. Run Ablations
```bash
python scripts/run_ablations.py --dataset hotpotqa --provider ollama
```

## 17. Run Evaluation
```bash
python scripts/run_evaluation.py --results-dir outputs/run_<timestamp>
```

## 18. Run All Experiments
```bash
python scripts/run_all_experiments.py --provider ollama
```

## 19. Verify Environment
```bash
python scripts/verify_environment.py
```

## 20. Reproducibility
Fixed seeds, pinned dependencies, deterministic candidate selection, and frozen retrieval indexes (`RAG_Project`).

## 21. Known Limitations
- Generative backends vary in their precise tokenization logic, which can result in minor deviations in the exact prompt size budget.
- Local LLM generation requires Ollama service running on `http://localhost:11434`.
- Mistral API limitations based on rate/tier could stall the pipeline.

## Project Structure
```
├── README.md
├── LICENSE
├── configs/
├── data/
├── docs/
├── notebooks/
├── outputs/
├── references/
├── scripts/
├── src/
└── tests/
```

## Hyperparameters (Validated System Defaults)
| Parameter | Default | Description |
|-----------|---------|-------------|
| alpha (α) | 0.7 | Semantic relevance weight |
| beta (β) | 0.3 | Evidential relevance weight |
| tau (τ) | 0.85 | Similarity threshold for cross-document redundancy |
| Token Budget | 300 | Maximum compressed context token budget |
| Retrieval Backend | dense | Dense FAISS index with `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval Top-K | 10 | Candidate documents retrieved per query |
