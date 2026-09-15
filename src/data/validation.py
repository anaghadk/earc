import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass

from src.data.schemas import QAExample

logger = logging.getLogger(__name__)

@dataclass
class ValidationReport:
    total: int
    valid: int
    invalid: int
    issues: Dict[str, int]

def validate_example(example: QAExample) -> Tuple[bool, List[str]]:
    issues = []
    if not example.question or not example.question.strip():
        issues.append("empty_question")
        
    if not example.answers:
        issues.append("no_answers")
    else:
        empty_answers = [a for a in example.answers if not a.strip()]
        if len(empty_answers) == len(example.answers):
            issues.append("all_answers_empty")
            
    is_valid = len(issues) == 0
    return is_valid, issues

def validate_dataset(examples: List[QAExample], dataset_name: str) -> ValidationReport:
    total = len(examples)
    valid = 0
    invalid = 0
    issues_counts = {}
    
    for ex in examples:
        is_valid, issues = validate_example(ex)
        if is_valid:
            valid += 1
        else:
            invalid += 1
            for issue in issues:
                issues_counts[issue] = issues_counts.get(issue, 0) + 1
                
    report = ValidationReport(
        total=total,
        valid=valid,
        invalid=invalid,
        issues=issues_counts
    )
    
    logger.info(f"Validation report for {dataset_name}: Total={total}, Valid={valid}, Invalid={invalid}")
    if invalid > 0:
        logger.warning(f"Issues found: {issues_counts}")
        
    return report

def check_no_answer_leakage(example: QAExample, compression_inputs: List[str]) -> bool:
    for answer in example.answers:
        ans_lower = answer.lower()
        for ci in compression_inputs:
            if ans_lower in ci.lower():
                return False
    return True
