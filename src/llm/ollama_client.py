import httpx
import time
import logging
import json
import tiktoken
from typing import Optional, Dict, Any

from src.llm.base import LLMProvider, TokenCounter
from src.data.schemas import GenerationResult

logger = logging.getLogger(__name__)

class OllamaTokenCounter(TokenCounter):
    """
    Token counter for Ollama models.
    Uses tiktoken with 'cl100k_base' encoding as an approximation for LLaMA and other open models,
    as exact tokenizers per model are not always available out of the box in python.
    """
    def __init__(self, model_name: str):
        self._model_name = model_name
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    @property
    def name(self) -> str:
        return f"tiktoken-cl100k_base (approx for {self._model_name})"

class OllamaLLMProvider(LLMProvider):
    def __init__(
        self,
        base_url: str = 'http://localhost:11434',
        model: str = 'llama3.2',
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def get_token_counter(self) -> TokenCounter:
        return OllamaTokenCounter(self._model)

    def check_availability(self) -> bool:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                for m in models:
                    if m.get("name") == self._model or m.get("name", "").startswith(self._model + ":"):
                        return True
                logger.warning(f"Model {self._model} not found in Ollama at {self.base_url}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            return False

    def generate(self, prompt: str) -> GenerationResult:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            start_time = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/api/generate",
                        json=payload
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    text = data.get("response", "")
                    prompt_tokens = data.get("prompt_eval_count")
                    completion_tokens = data.get("eval_count")
                    
                    if prompt_tokens is None or completion_tokens is None:
                        counter = self.get_token_counter()
                        prompt_tokens = counter.count(prompt)
                        completion_tokens = counter.count(text)
                        
                    total_tokens = prompt_tokens + completion_tokens
                    latency = time.perf_counter() - start_time
                    
                    logger.info(f"Ollama generate: model={self._model}, latency={latency:.2f}s, tokens={total_tokens}")
                    
                    return GenerationResult(
                        text=text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        latency=latency,
                        model=self._model,
                        provider=self.provider_name
                    )
            except httpx.HTTPStatusError as e:
                # Retry on 5xx errors
                last_error = e
                if e.response.status_code >= 500:
                    delay = 2 ** attempt
                    logger.warning(f"Ollama server error (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # Client errors (4xx) shouldn't be retried
                    raise ValueError(f"Ollama API error: {e.response.text}") from e
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, json.JSONDecodeError) as e:
                last_error = e
                delay = 2 ** attempt
                logger.warning(f"Ollama transient error (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)

        raise RuntimeError(f"Ollama generate failed after {self.max_retries} retries. Last error: {last_error}")
