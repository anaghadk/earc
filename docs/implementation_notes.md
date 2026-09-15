# Implementation Notes

This document highlights the deviations and scale-ups from the original EARC paper to our implementation.

## 1. Scale Up: 300 -> 5,000 Examples
- **Paper**: Evaluated the method on 300 test instances per dataset.
- **Implementation**: Scaled to **5,000 examples per dataset**, providing robust evaluation and testing the pipeline under realistic computational load.

## 2. Dataset Expansion: NQ Added
- **Paper**: Evaluated on HotpotQA and TriviaQA. Mentioned exploring other datasets as future work.
- **Implementation**: Incorporated the **Natural Questions (NQ)** dataset, scaling the benchmark to 3 distinct QA setups and testing the versatility of EARC in single-hop and multi-hop scenarios.

## 3. Generative Model Backend: Flan-T5 -> Ollama + Mistral
- **Paper**: Relied on Flan-T5-base (an encoder-decoder architecture) for QA generation.
- **Implementation**: Utilizes modern autoregressive decoder-only models.
    - **Ollama**: Running `llama3.2` locally.
    - **Mistral API**: As a cloud-hosted LLM baseline.
  This introduces differences in token counting since the LLM backends employ different tokenizers compared to T5.

## 4. Hardware Assumptions
- **Paper**: Assumes specific compute availability.
- **Implementation**: We assume CUDA 12.1+ compatible hardware, given the large dataset size, but we ensure graceful failure or reasonable CPU-fallback where strict GPU is unavailable, though GPU is strongly recommended.

## 5. Token Budget Handling
- **Paper**: Token calculations were specific to T5 tokenization.
- **Implementation**: Handled generically via LLM-specific adapter classes (`TokenCounter` abstraction in `src/llm/base.py`). This guarantees that we strictly respect token budgets regardless of the chosen LLM backend (Ollama vs Mistral).

## Reference: Original Numbers (Tables II-VI)
The paper (Tables II-VI) demonstrates superior F1 and ROUGE scores at compression ratios of 300-400 tokens compared to vanilla Retrieval and standard Summarization techniques. Our goal with these 15,000 examples is to re-verify if those performance margins translate to larger test splits and different LLMs.
