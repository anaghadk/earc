import logging
import re
from typing import List, Set, Optional, Tuple
from src.data.schemas import CandidateSentence

logger = logging.getLogger(__name__)

DEFAULT_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
    'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
    "won't", 'wouldn', "wouldn't"
}

# ---------------------------------------------------------------------------
# Question-type evidence bonus
#
# Each rule is a (question_trigger, sentence_context_marker, bonus) triple.
# BOTH patterns must match for the bonus to fire.
#
# The sentence_context_marker uses SEMANTIC RELATIONSHIP words (locative
# phrases, role-introduction verbs, noun frames for creative works) rather
# than surface-form heuristics like capitalisation.  This prevents the
# distractor-boosting problem observed with spaCy NER in experiments.
# ---------------------------------------------------------------------------
_QTYPE_RULES: List[Tuple[re.Pattern, re.Pattern, int]] = [

    # WHO / WHICH <role noun>
    # Sentence marker: passive role-introduction phrase ("directed by",
    # "voiced by", "born in", "known as", ...).  Rare in distractor sentences
    # that describe their own subject without attributing it to a person.
    (
        re.compile(
            r'\bwho\b'
            r'|\bwhich\s+(actor|actress|director|singer|musician|pianist'
            r'|composer|writer|author|player|performer|founder|producer|host)\b',
            re.I,
        ),
        re.compile(
            r'\b(directed by|written by|composed by|founded by|performed by'
            r'|played by|hosted by|produced by|voiced by'
            r'|born in|known as|served as|appointed as|rose to fame)\b',
            re.I,
        ),
        2,
    ),

    # WHEN / WHAT YEAR
    # Sentence marker: 4-digit year in range 1000-2029.  Year numbers are
    # highly specific; they rarely appear in distractor sentences that do
    # not describe temporal events.
    (
        re.compile(r'\bwhen\b|\bwhat year\b|\bin what year\b', re.I),
        re.compile(r'\b(1[0-9]{3}|20[0-2][0-9])\b'),
        2,
    ),

    # HOW MANY / HOW MUCH / HOW LONG / HOW OLD
    # Sentence marker: any cardinal number.  The question explicitly expects
    # a numeric answer, so numeric sentences are almost always more relevant.
    (
        re.compile(r'\bhow\s+(many|much|long|old|far|tall|large|big)\b', re.I),
        re.compile(r'\b\d[\d,]*(?:\.\d+)?\b'),
        2,
    ),

    # WHAT COUNTRY / WHERE / WHICH COUNTRY|CITY|CONTINENT
    # Sentence marker: explicit LOCATIVE RELATIONSHIP PHRASE.
    # "based in", "located in", "found in", "native to", "grows in", etc.
    # These phrases appear almost exclusively in sentences that state a
    # geographic location.  Generic sentences ("Bernstein composed...",
    # "The 1957 Tony Award...") do not contain them.
    (
        re.compile(
            r'\bwhat\s+(country|city|town|region|state|continent)\b'
            r'|\bwhere\b'
            r'|\bin what\s+(country|city|town|region|state|continent)\b',
            re.I,
        ),
        re.compile(
            r'\b(based in|located in|found in|native to|situated in'
            r'|grows?\s+in|lives?\s+in|born in|headquartered in'
            r'|originates?\s+from|endemic to|distributed\s+(across|in|throughout))\b',
            re.I,
        ),
        2,
    ),

    # WHAT PLAY / FILM / BOOK / SONG / ALBUM (creative work)
    # Sentence marker: the same work-type word appears as a NOUN in the
    # sentence, identified by one of four frames:
    #   (a) article/possessive + work-type noun  ("the play", "his song")
    #   (b) possessive proper noun + work-type   ("Shakespeare's play")
    #   (c) work-type noun + title introducer    ("play called", "film titled")
    #   (d) work-type noun + opening quote       ("play 'Romeo...")
    # Whole-word \bplay\b does NOT match the verb "plays", so
    # "Dag Achatz plays piano" is safe.
    (
        re.compile(
            r'\bwhat\s+(play|musical|film|movie|book|novel|song|album'
            r'|opera|ballet|show|series|episode|composition)\b'
            r'|\bwhich\s+(play|musical|film|movie|book|novel|song|album'
            r'|opera|ballet|show|series|episode|composition)\b',
            re.I,
        ),
        re.compile(
            # (a) article/possessive determiner + work-type noun
            r'\b(?:a|an|the|his|her|their|its|your)\s+'
            r'(?:play|musical|film|movie|book|novel|song|album'
            r'|opera|ballet|show|series|episode|composition)\b'
            # (b) possessive proper noun + work-type noun
            r"|[A-Z][a-z]+'s\s+(?:play|musical|film|movie|book|novel|song|album"
            r'|opera|ballet|show|series|episode|composition)\b'
            # (c) work-type noun + title introducer keyword
            r'|\b(?:play|musical|film|movie|book|novel|song|album'
            r'|opera|ballet|show|series|episode|composition)'
            r'\s+(?:called|titled|named|entitled)\b'
            # (d) work-type noun immediately followed by an opening quote
            r'|\b(?:play|musical|film|movie|book|novel|song|album'
            r"|opera|ballet|show|series|episode|composition)\s+[\"']",
            re.I,
        ),
        2,
    ),

    # PART / REGION / COMPONENT ("which part of", "what region of")
    # Sentence marker: definitional / partitive copula ("is a part of", "is a region of", ...)
    (
        re.compile(
            r'\b(?:which|what)\s+(?:part|region|component|organ|area|section)\s+of\b',
            re.I,
        ),
        re.compile(
            r'\b(?:is|are)\s+(?:a|an|the)?\s*(?:part|region|structure|organ|area|division|component|portion)\s+of\b',
            re.I,
        ),
        2,
    ),

    # SPORT / GAME DOMAIN ("which sport", "what sport")
    # Sentence marker: sport glossary / sport name context
    (
        re.compile(
            r'\b(?:which|what)\s+(?:sport|game)\b',
            re.I,
        ),
        re.compile(
            r'\b(?:rugby|football|cricket|tennis|basketball|baseball|hockey|golf|soccer|sport|game)\b',
            re.I,
        ),
        2,
    ),
]


_DESCRIPTIVE_EVIDENCE_PATTERN = re.compile(
    r"\b(process|mechanism|convert|converts|converting|conversion|transformation|"
    r"reaction|reactions|leads\s+to|results?\s+in|caused\s+by|because|due\s+to|"
    r"as\s+a\s+result|consequence|consequences|reason\s+for|reasons?\s+for|"
    r"factors?\s+that|collapse|collapsed|decline|declined|decay|downfall|fell|"
    r"fall\s+of|works?\s+by|functioning|pathway|cycle|producing|produces|synthesis|"
    r"synthesize|cellular|breakdown|origin|involve|involves)\b",
    re.IGNORECASE,
)

_TEMPORAL_MARKER = re.compile(
    r"\b(1[0-9]{3}|20[0-2][0-9]|\d{1,2}/\d{1,2}/\d{2,4}|january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_LOCATIVE_MARKER = re.compile(
    r"\b(based in|located in|found in|native to|situated in|born in|headquartered in|"
    r"originates?\s+from|birthplace|place of birth|native of|citizen of|nationality|"
    r"serbia|serbian|croatia|croatian|austrian|austria|france|french|italy|italian|germany|german|america|american)\b",
    re.IGNORECASE,
)
_PERSON_MARKER = re.compile(
    r"\b(directed by|written by|composed by|founded by|performed by|played by|hosted by|"
    r"produced by|voiced by|invented by|established by|started by)\b",
    re.IGNORECASE,
)


def _detect_qtype_bonus(query: str, text: str) -> int:
    """Return +2 if the sentence matches the expected answer type, else 0.

    Requires BOTH the question trigger AND the sentence-level semantic
    context marker to match.  Only the first matching rule fires.
    No oracle labels, no external models, no new dependencies.
    """
    for q_pattern, s_pattern, bonus in _QTYPE_RULES:
        if q_pattern.search(query) and s_pattern.search(text):
            return bonus
    return 0


class EvidenceScorer:
    def __init__(self, spacy_model: str = 'en_core_web_sm', stopwords: Optional[Set[str]] = None):
        self.spacy_model = spacy_model
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self.nlp = None
        self.num_pattern = re.compile(r'\b\d+(?:\.\d+)?%?\b|\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b')

    def _load_model(self):
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load(self.spacy_model)
            except ImportError:
                logger.warning(f"spaCy not installed or model {self.spacy_model} not found. NER will be skipped.")
            except OSError:
                logger.warning(f"spaCy model {self.spacy_model} not found. NER will be skipped.")

    def _tokenize(self, text: str) -> Set[str]:
        tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
        return {t for t in tokens if t and t not in self.stopwords}

    def score(
        self,
        candidates: List[CandidateSentence],
        query: str,
        q_type: Optional[str] = None,
        q_details: Optional[dict] = None,
    ) -> List[CandidateSentence]:
        # NER entity counting is intentionally disabled: spaCy NER boosts
        # entity-dense sentences indiscriminately, which hurt overall EM
        # in experiments (65% -> 60%).  entity_count stays 0; scoring relies
        # solely on number matches and query-keyword overlap.
        from src.compression.query_classification import classify_question

        if q_type is None or q_details is None:
            detected_type, detected_details = classify_question(query)
            q_type = q_type or detected_type
            q_details = q_details or detected_details

        query_tokens = self._tokenize(query)
        is_descriptive = (q_type == "DESCRIPTIVE")
        is_multipart = bool(q_details.get("is_multipart", False))
        req_attrs = q_details.get("requested_attributes", set())

        for candidate in candidates:
            entity_count = 0
                
            number_matches = self.num_pattern.findall(candidate.text)
            number_count = len(number_matches)
            
            # Title-aware contextual text for keyword hits and QType matching
            combined_text = f"{candidate.title}: {candidate.text}" if (candidate.title and candidate.title.lower() not in candidate.text.lower()) else candidate.text

            sentence_tokens = self._tokenize(combined_text)
            keyword_hits = len(query_tokens & sentence_tokens)
            
            candidate.entity_count = entity_count
            candidate.number_count = number_count
            candidate.keyword_hits = keyword_hits
            
            candidate.evidence_score = float(entity_count + number_count + keyword_hits)

            # Document title exact match with query entities (e.g. "Arthur's Magazine" in query)
            if candidate.title and len(candidate.title) > 3 and candidate.title.lower() in query.lower():
                candidate.evidence_score += 2.0

            if is_descriptive:
                # Prioritize explanatory evidence: mechanism, process, cause, reason, effect
                if _DESCRIPTIVE_EVIDENCE_PATTERN.search(combined_text):
                    candidate.evidence_score += 3.0
            else:
                # Standard factoid / multi-hop answer-bearing markers
                candidate.evidence_score += _detect_qtype_bonus(query, combined_text)

                # For multi-part factoids, boost sentences that cover each requested facet
                if is_multipart:
                    if "temporal" in req_attrs and _TEMPORAL_MARKER.search(combined_text):
                        candidate.evidence_score += 2.0
                    if "locative" in req_attrs and _LOCATIVE_MARKER.search(combined_text):
                        candidate.evidence_score += 2.0
                    if "person" in req_attrs and _PERSON_MARKER.search(combined_text):
                        candidate.evidence_score += 2.0
            
        return candidates

    def normalize(self, candidates: List[CandidateSentence]) -> List[CandidateSentence]:
        if not candidates:
            return candidates
            
        scores = [c.evidence_score for c in candidates if c.evidence_score is not None]
        if not scores:
            return candidates
            
        min_score = min(scores)
        max_score = max(scores)
        
        for candidate in candidates:
            if candidate.evidence_score is None:
                candidate.normalized_evidence = 0.0
            elif max_score == min_score:
                candidate.normalized_evidence = 0.0
            else:
                candidate.normalized_evidence = (candidate.evidence_score - min_score) / (max_score - min_score)
                
        return candidates
