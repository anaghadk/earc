import time
import logging
import random
import tiktoken
from typing import Optional, Dict, Any
from mistralai.client import Mistral

from src.llm.base import LLMProvider, TokenCounter
from src.data.schemas import GenerationResult

logger = logging.getLogger(__name__)

class MistralTokenCounter(TokenCounter):
    """
    Token counter for Mistral models.
    Uses tiktoken with 'cl100k_base' encoding as an approximation.
    """
    def __init__(self, model_name: str):
        self._model_name = model_name
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    @property
    def name(self) -> str:
        return f"tiktoken-cl100k_base (approx for {self._model_name})"

class MistralLLMProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = 'mistral-small-latest',
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: int = 120,
        max_retries: int = 3
    ):
        if not api_key:
            raise ValueError("Mistral API key must be provided")
            
        self.api_key = api_key
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "mistral"

    @property
    def model_name(self) -> str:
        return self._model

    def get_token_counter(self) -> TokenCounter:
        return MistralTokenCounter(self._model)

    def check_availability(self) -> bool:
        if not self.api_key:
            return False
        try:
            client = Mistral(api_key=self.api_key)
            models_response = client.models.list()
            return True
        except Exception as e:
            logger.error(f"Mistral check_availability failed: {e}")
            return False

    def generate(self, prompt: str) -> GenerationResult:
        client = Mistral(api_key=self.api_key)
        
        last_error = None
        for attempt in range(self.max_retries):
            start_time = time.perf_counter()
            try:
                response = client.chat.complete(
                    model=self._model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                text = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                latency = time.perf_counter() - start_time
                
                logger.info(f"Mistral generate: model={self._model}, latency={latency:.2f}s, tokens={total_tokens}")
                
                return GenerationResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency=latency,
                    model=self._model,
                    provider=self.provider_name
                )
            except Exception as e:
                error_msg = str(e).lower()
                is_retryable = (
                    "429" in error_msg or 
                    "timeout" in error_msg or 
                    "50" in error_msg or 
                    "rate limit" in error_msg
                )
                
                if "401" in error_msg or "unauthorized" in error_msg:
                    logger.error("Mistral API authentication failed (401). Check API key.")
                    raise RuntimeError("Mistral API authentication failed") from e
                
                if is_retryable:
                    last_error = e
                    # Exponential backoff with jitter
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Mistral retryable error (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Mistral API error: {e}") from e

        raise RuntimeError(f"Mistral generate failed after {self.max_retries} retries. Last error: {last_error}")
