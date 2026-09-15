import logging
import pickle
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset as hf_load_dataset

from src.data.schemas import QAExample, Document

logger = logging.getLogger(__name__)

class NQLoader:
    def load(self, split: str = 'train', cache_dir: Optional[str] = None) -> List[QAExample]:
        logger.info(f"Loading nq_open dataset, split: {split}")
        try:
            dataset = hf_load_dataset('google-research-datasets/nq_open', split=split, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to load nq_open: {e}")
            raise

        examples = []
        for item in dataset:
            question = item['question'].strip()
            answers = [a.strip() for a in item['answer']]
            example = QAExample(
                id=question,  # NQ doesn't have a unique ID by default
                dataset="nq",
                question=question,
                answers=answers,
                documents=[]
            )
            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from nq_open")
        return examples

class HotpotQALoader:
    def load(self, split: str = 'train', cache_dir: Optional[str] = None) -> List[QAExample]:
        logger.info(f"Loading hotpot_qa (distractor) dataset, split: {split}")
        try:
            dataset = hf_load_dataset('hotpotqa/hotpot_qa', 'distractor', split=split, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to load hotpot_qa: {e}")
            raise

        examples = []
        for item in dataset:
            question = item['question'].strip()
            answer = item['answer'].strip()
            
            supporting_titles = set(item['supporting_facts']['title'])
            documents = []
            
            context_titles = item['context']['title']
            context_sentences = item['context']['sentences']
            
            for title, sentences in zip(context_titles, context_sentences):
                text = " ".join(sentences)
                is_supporting = title in supporting_titles
                doc = Document(
                    doc_id=title,
                    title=title,
                    text=text,
                    is_supporting=is_supporting
                )
                documents.append(doc)
            
            example = QAExample(
                id=item['id'],
                dataset="hotpotqa",
                question=question,
                answers=[answer],
                documents=documents
            )
            examples.append(example)

        logger.info(f"Loaded {len(examples)} examples from hotpot_qa")
        return examples

class TriviaQALoader:
    def load(self, split: str = 'train', cache_dir: Optional[str] = None) -> List[QAExample]:
        logger.info(f"Loading trivia_qa dataset, split: {split}")

        # 1. Prefer pre-extracted QA pairs aligned with RAG_Project corpus
        rag_qa_dir = Path("RAG_Project/qa_pairs")
        if rag_qa_dir.exists():
            pkl_files = sorted(rag_qa_dir.glob("qa_*.pkl"))
            if pkl_files:
                examples = []
                for pkl_file in pkl_files:
                    with open(pkl_file, "rb") as f:
                        items = pickle.load(f)
                    for item in items:
                        if item.get("dataset") == "trivia":
                            q = item["question"].strip()
                            ans = [a.strip() for a in item.get("answers", []) if a.strip()]
                            examples.append(QAExample(
                                id=item.get("question_id", q),
                                dataset="triviaqa",
                                question=q,
                                answers=ans,
                                documents=[]
                            ))
                logger.info(f"Loaded {len(examples)} TriviaQA examples from {rag_qa_dir}")
                return examples

        # 2. Fallback to HuggingFace
        try:
            dataset = hf_load_dataset('mandarjoshi/trivia_qa', 'rc', split=split, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to load trivia_qa: {e}")
            raise

        examples = []
        for item in dataset:
            question = item['question'].strip()
            answer_dict = item['answer']
            answers = [answer_dict['value']] + answer_dict.get('aliases', [])
            answers = list(set([a.strip() for a in answers if a.strip()]))
            
            documents = []
            if 'entity_pages' in item and item['entity_pages']:
                entity_pages = item['entity_pages']
                titles = entity_pages.get('title', [])
                texts = entity_pages.get('wiki_context', [])
                filenames = entity_pages.get('filename', [])
                for title, text, fname in zip(titles, texts, filenames):
                    if text.strip():
                        doc = Document(
                            doc_id=fname if fname else title,
                            title=title,
                            text=text,
                            is_supporting=False
                        )
                        documents.append(doc)
            elif 'search_results' in item and item['search_results']:
                search_results = item['search_results']
                titles = search_results.get('title', [])
                texts = search_results.get('search_context', [])
                filenames = search_results.get('filename', [])
                for title, text, fname in zip(titles, texts, filenames):
                    if text.strip():
                        doc = Document(
                            doc_id=fname if fname else title,
                            title=title,
                            text=text,
                            is_supporting=False
                        )
                        documents.append(doc)

            example = QAExample(
                id=item['question_id'],
                dataset="triviaqa",
                question=question,
                answers=answers,
                documents=documents
            )
            examples.append(example)

        logger.info(f"Loaded {len(examples)} examples from trivia_qa")
        return examples

def load_dataset(name: str, split: str = 'train', cache_dir: Optional[str] = None) -> List[QAExample]:
    name = name.lower()
    if name in ['nq', 'nq_open']:
        return NQLoader().load(split, cache_dir)
    elif name in ['hotpotqa', 'hotpot_qa']:
        return HotpotQALoader().load(split, cache_dir)
    elif name in ['triviaqa', 'trivia_qa']:
        return TriviaQALoader().load(split, cache_dir)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def load_nq_passage_pool(n_passages: int, cache_dir: Optional[str] = None) -> List[Document]:
    """
    NQ Open ships no context, so (matching the earlier validated pipeline)
    pull real passages from the companion sentence-transformers/natural-questions
    dataset. That dataset stores the passage text under the 'answer' column
    (a naming quirk of its own, not a short answer). These passages are NOT
    matched 1:1 to specific NQ questions -- they go into the shared retrieval
    corpus alongside HotpotQA/TriviaQA documents, exactly like the old pipeline.
    """
    logger.info(f"Loading {n_passages} NQ passages from sentence-transformers/natural-questions")
    try:
        dataset = hf_load_dataset(
            'sentence-transformers/natural-questions', split='train', cache_dir=cache_dir
        )
    except Exception as e:
        logger.error(f"Failed to load sentence-transformers/natural-questions: {e}")
        raise

    if n_passages and n_passages < len(dataset):
        dataset = dataset.select(range(n_passages))

    passages = []
    seen_prefixes = set()
    for idx, row in enumerate(dataset):
        text = str(row.get('answer', '') or '').strip()
        if not text:
            continue
        dedup_key = text[:200]
        if dedup_key in seen_prefixes:
            continue
        seen_prefixes.add(dedup_key)
        passages.append(Document(
            doc_id=f"nq_passages_{idx}",
            title="",
            text=text,
        ))

    logger.info(f"Loaded {len(passages)} deduplicated NQ passages")
    return passages
