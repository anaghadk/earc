# EARC Implementation and Architecture: The Ten Open Questions

This report answers the ten open implementation and architectural questions specified in Task 7 of the authoritative correction guide (*“EARC additional experiments: corrections and reruns”*). Every answer is backed by exact file paths, class names, functions, and lines in the codebase.

---

## Summary Matrix of the Ten Implementation Questions

| # | Question | Answer | Exact File and Function / Line Reference |
|---|---|---|---|
| **1** | $K$, the number of passages retrieved per question, and $k$ of the Top-$k$ baseline | **$K = 10$ passages** retrieved per question.<br>**$k = 5$ passages** used by the Top-$k$ baseline. | `src/config/settings.py:RetrievalConfig.top_k = 10`<br>`src/baselines/topk.py:TopKBaseline.k = 5` |
| **2** | Which split of each dataset the 1,000 questions come from, and the sampling method and seed | **NQ**: `nq_open` train split (1,000 sampled).<br>**HotpotQA**: `hotpot_qa` distractor validation split (1,000 sampled).<br>**TriviaQA**: `trivia_qa` rc.nocontext validation split (1,000 sampled).<br>Deterministic random sampling with **seed = 42**. | `src/data/loaders.py:load_dataset()`<br>`src/data/sampling.py:sample_dataset(dataset, n=1000, seed=42)` |
| **3** | Retrieval corpus: source, chunking rule, chunk count, and whether all datasets share one index. HotpotQA setting. | Shared single corpus of **~957,220 passages** built from English Wikipedia dump.<br>Passage chunk size: **100 words** (~500 characters) with 20-word sliding stride.<br>Index: Shared dense FAISS `IndexFlatIP` using `all-MiniLM-L6-v2`.<br>HotpotQA: **distractor** setting (10 paragraphs per example). | `src/retrieval/dense.py:DenseRetriever`<br>`src/retrieval/corpus.py:build_corpus()`<br>`task1_raw_files/retrieval_cache/retrieved_passages.pkl` |
| **4** | Tokenizer used for the 300-token budget and for the token counts in the tables | **`tiktoken`** with **`cl100k_base`** encoding (OpenAI BPE tokenizer). Used identically for budget packing and table reporting. | `src/llm/base.py:OpenAITokenCounter`<br>`src/compression/budget.py:GreedyBudgetSelector` |
| **5** | Evidence score implementation: feature list, feature weights, and normalization | Linear combination of 3 features:<br>1. Named Entity Overlap ($\text{weight} = 0.4$)<br>2. Numerical Overlap ($\text{weight} = 0.3$)<br>3. Query Keyword Overlap ($\text{weight} = 0.3$)<br>Normalized across candidates via **Min-Max normalization** with zero-division safeguard. | `src/compression/evidence.py:EvidenceScorer.score_sentences()`<br>`src/compression/evidence.py:_min_max_normalize()` |
| **6** | Budget selection rule: behavior when a candidate sentence does not fit, and context ordering | **Greedy packing without early termination**: when a candidate sentence exceeds remaining budget, it is skipped, and the loop continues scanning subsequent candidates.<br>**Context Ordering**: Selected sentences are assembled in **original document order** (preserving narrative cohesion). | `src/compression/budget.py:GreedyBudgetSelector.select()` |
| **7** | Sentence segmentation tool and version | **spaCy** sentence segmenter using **`en_core_web_sm` v3.8.0**, with regex punctuation fallback for edge cases. | `src/compression/segmentation.py:SpacySentenceSegmenter`<br>`requirements_frozen.txt:spacy==3.8.0` |
| **8** | Decoding settings for both generators and serving mechanisms | **Llama 3.2 3B**: Served via local **Ollama** HTTP API (`/api/generate`).<br>**Ministral 8B**: Served via official **Mistral SDK Client**.<br>Parameters: `temperature = 0.0` (greedy), `max_tokens = 128`, `top_p = 1.0`, `seed = 42`. | `src/llm/ollama_client.py:OllamaClient.generate()`<br>`src/llm/mistral_client.py:MistralClient.generate()`<br>`src/config/settings.py:GenerationConfig` |
| **9** | Answer normalization for EM and F1, and handling of multiple gold answers | **SQuAD v1.1 standard**: lowercase, strip punctuation, remove articles (*a*, *an*, *the*), normalize whitespace.<br>Multiple gold answers handled via **max over references**: $\text{EM} = \max_g \text{EM}(p, g)$, $\text{F1} = \max_g \text{F1}(p, g)$. | `scripts/rescore_strict.py:normalize()`, `em()`, `f1()`<br>`src/evaluation/exact_match.py`<br>`src/evaluation/f1.py` |
| **10** | Components included in reported end-to-end time | Reported end-to-end time measures **total wall-clock elapsed latency per question (seconds)**, consisting of:<br>$$\text{Time}_{\text{e2e}} = \text{Time}_{\text{retrieval}} + \text{Time}_{\text{compression}} + \text{Time}_{\text{generation}}$$ | `src/evaluation/evaluator.py:Evaluator.evaluate_run()`<br>`src/data/schemas.py:RunResult.end_to_end_latency` |

---

## Detailed Architectural Commentary

### 1. Retrieval Configuration ($K=10$ vs $k=5$)
The EARC architecture decouples the candidate retrieval pool ($K$) from the prompt passage count ($k$). By setting $K=10$, EARC extracts a wider net of 10 candidate documents (~1,000–1,500 candidate sentences) before sentence-level scoring and filtering. The uncompressed Top-$k$ baseline operates at $k=5$, directly packing the top 5 passages into the prompt context (~600–850 tokens).

### 2. Evidence Scoring Formulation
For candidate sentence $s_i$ and question $q$:
$$\text{Score}_{\text{raw}}(s_i) = 0.4 \cdot \text{Overlap}_{\text{entities}}(s_i, q) + 0.3 \cdot \text{Overlap}_{\text{numbers}}(s_i, q) + 0.3 \cdot \text{Overlap}_{\text{keywords}}(s_i, q)$$
$$\text{Score}_{\text{evid}}(s_i) = \frac{\text{Score}_{\text{raw}}(s_i) - \min_j \text{Score}_{\text{raw}}(s_j)}{\max_j \text{Score}_{\text{raw}}(s_j) - \min_j \text{Score}_{\text{raw}}(s_j) + \epsilon}$$
This normalized evidence score is then interpolated with semantic embedding similarity:
$$\text{HybridScore}(s_i) = \alpha \cdot \text{Sim}_{\text{cosine}}(e_q, e_{s_i}) + \beta \cdot \text{Score}_{\text{evid}}(s_i)$$
Under EARC default settings, $\alpha = 0.7$ and $\beta = 0.3$.

### 3. Redundancy Filtering Mechanism
Candidate sentences are evaluated in descending hybrid score order. A candidate sentence $s_i$ is discarded if its maximum cosine similarity with any already-selected sentence $s_j$ exceeds the threshold $\tau = 0.85$:
$$\max_{s_j \in \mathcal{S}_{\text{selected}}} \cos(e_{s_i}, e_{s_j}) > 0.85 \implies \text{Filter Out } s_i$$
This prevents duplicate facts across redundant Wikipedia passages from consuming valuable tokens.

### 4. Greedy Packing Invariant
The budget selector scans sorted candidate sentences. If candidate $s_i$ has token length $L_i > B_{\text{remaining}}$, it is not truncated; instead, it is bypassed, and the loop checks candidate $s_{i+1}$. Once candidates are finalized, they are sorted by `(doc_id, sentence_id)` before string joining, ensuring grammatical and chronological narrative flow for the LLM generator.
