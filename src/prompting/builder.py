import logging
from src.prompting.templates import DEFAULT_QA_TEMPLATE
from src.data.schemas import CompressedContext

logger = logging.getLogger(__name__)

class PromptBuilder:
    def __init__(self, template: str = DEFAULT_QA_TEMPLATE):
        """
        Initialize the prompt builder with a given template.
        
        Args:
            template: The prompt template string containing '{question}' and '{context}' placeholders.
        """
        self.template = template
        if "{question}" not in self.template or "{context}" not in self.template:
            raise ValueError("Template must contain both '{question}' and '{context}' placeholders.")

    def build(
        self,
        question: str,
        context: str,
        q_type: str | None = None,
        q_details: dict | None = None,
    ) -> str:
        """
        Build a prompt by formatting the template with the provided question and context.
        Uses question-type specific instructions when appropriate.
        """
        from src.compression.query_classification import classify_question
        from src.prompting.templates import (
            DEFAULT_QA_TEMPLATE,
            DESCRIPTIVE_QA_TEMPLATE,
            MULTIHOP_QA_TEMPLATE,
            MULTIPART_FACTOID_QA_TEMPLATE,
        )

        if q_type is None or q_details is None:
            detected_type, detected_details = classify_question(question)
            q_type = q_type or detected_type
            q_details = q_details or detected_details

        if q_type == "DESCRIPTIVE":
            tmpl = DESCRIPTIVE_QA_TEMPLATE
        elif q_type == "MULTI-HOP":
            tmpl = MULTIHOP_QA_TEMPLATE
        elif q_type == "FACTOID" and q_details and q_details.get("is_multipart"):
            tmpl = MULTIPART_FACTOID_QA_TEMPLATE
        else:
            tmpl = self.template or DEFAULT_QA_TEMPLATE

        return tmpl.format(question=question, context=context)

    def build_from_compressed(
        self,
        question: str,
        compressed: CompressedContext,
        q_type: str | None = None,
        q_details: dict | None = None,
    ) -> str:
        """
        Build a prompt directly from a CompressedContext object.
        """
        return self.build(
            question=question,
            context=compressed.compressed_text,
            q_type=q_type,
            q_details=q_details,
        )

    def validate_prompt(self, prompt: str, question: str) -> bool:
        """
        Validate a constructed prompt to ensure it meets requirements:
        1. Contains the exact question.
        2. Doesn't leak gold answers or specific ranking scores.
        
        Args:
            prompt: The generated prompt.
            question: The original question.
            
        Returns:
            True if valid, False otherwise.
        """
        if question.strip() not in prompt:
            logger.warning("Validation failed: Question not found in prompt.")
            return False
            
        forbidden_keywords = [
            "gold_answer",
            "evidence_label",
            "semantic_score=",
            "evidence_score=",
            "hybrid_score="
        ]
        
        for keyword in forbidden_keywords:
            if keyword in prompt.lower():
                logger.warning(f"Validation failed: Forbidden keyword '{keyword}' found in prompt.")
                return False
                
        return True
