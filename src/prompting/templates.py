DEFAULT_QA_TEMPLATE = """You are answering a question using retrieved evidence.

Question:
{question}

Evidence:
{context}

Use only the provided evidence. Do not invent or assume information not present in the evidence.
The final answer should be concise — preferably the exact person, place, organization, object, or phrase that answers the question.

Respond in this exact format:
Final answer: <short answer>"""

FACTOID_QA_TEMPLATE = DEFAULT_QA_TEMPLATE

MULTIPART_FACTOID_QA_TEMPLATE = """You are answering a question using retrieved evidence.

Question:
{question}

Evidence:
{context}

Use only the provided evidence. Do not invent or assume information not present in the evidence.
The final answer should directly answer all requested parts based on the evidence. For questions asking "when and where", state both the birth year/date and birthplace/origin supported by the evidence. Never invent a location, and never discuss or deny locations absent from the evidence.

Respond in this exact format:
Final answer: <answer>"""

DESCRIPTIVE_QA_TEMPLATE = """You are answering a question using retrieved evidence.

Question:
{question}

Evidence:
{context}

Use only the provided evidence. Do not invent or assume information not present in the evidence.
Provide a clear, direct, and concise explanation of the process, mechanism, or causes described in the evidence.

Respond in this exact format:
Final answer: <explanation>"""

MULTIHOP_QA_TEMPLATE = """You are answering a question using retrieved evidence.

Question:
{question}

Evidence:
{context}

Use only the provided evidence. Do not invent or assume information not present in the evidence.
The final answer must be concise. If the question asks to choose between options (such as "Which ... A or B?"), output only the exact chosen option name from the question without extra words or commentary.

Respond in this exact format:
Final answer: <short answer>"""

TEMPLATE_REGISTRY = {
    'default': DEFAULT_QA_TEMPLATE,
    'factoid': FACTOID_QA_TEMPLATE,
    'multipart_factoid': MULTIPART_FACTOID_QA_TEMPLATE,
    'descriptive': DESCRIPTIVE_QA_TEMPLATE,
    'multihop': MULTIHOP_QA_TEMPLATE,
}

def get_template(name: str = 'default') -> str:
    """
    Retrieve a prompt template by name.
    
    Args:
        name: The name of the template in the registry.
        
    Returns:
        The format string for the template.
        
    Raises:
        KeyError: If the template name is not found in the registry.
    """
    if name not in TEMPLATE_REGISTRY:
        raise KeyError(f"Template '{name}' not found. Available templates: {list(TEMPLATE_REGISTRY.keys())}")
    return TEMPLATE_REGISTRY[name]
