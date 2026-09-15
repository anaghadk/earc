import os
from typing import Optional

from src.config.settings import EARCConfig
from src.llm.base import LLMProvider, TokenCounter
from src.llm.ollama_client import OllamaLLMProvider
from src.llm.mistral_client import MistralLLMProvider
from src.data.schemas import GenerationResult

class SimpleTokenCounter(TokenCounter):
    def count(self, text: str) -> int:
        return len(text.split())
        
    @property
    def name(self) -> str:
        return "simple_word_count"

class MockLLMProvider(LLMProvider):
    def __init__(self, model_name: str = "mock-model"):
        self._model = model_name
        
    @property
    def provider_name(self) -> str:
        return "mock"
        
    @property
    def model_name(self) -> str:
        return self._model
        
    def get_token_counter(self) -> TokenCounter:
        return SimpleTokenCounter()
        
    def check_availability(self) -> bool:
        return True
        
    def generate(self, prompt: str) -> GenerationResult:
        text = "mock answer"
        counter = self.get_token_counter()
        return GenerationResult(
            text=text,
            prompt_tokens=counter.count(prompt),
            completion_tokens=counter.count(text),
            total_tokens=counter.count(prompt) + counter.count(text),
            latency=0.01,
            model=self._model,
            provider=self.provider_name
        )

def create_provider(config: EARCConfig, provider_name: str) -> LLMProvider:
    provider_name = provider_name.lower()
    
    if provider_name == 'ollama':
        return OllamaLLMProvider(
            base_url=config.ollama.base_url,
            model=config.ollama.model,
            temperature=config.ollama.temperature,
            max_tokens=config.ollama.max_tokens,
            timeout=config.ollama.timeout,
            max_retries=config.ollama.max_retries
        )
    elif provider_name == 'mistral':
        api_key = config.mistral.api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API key not found in config or MISTRAL_API_KEY environment variable")
            
        return MistralLLMProvider(
            api_key=api_key,
            model=config.mistral.model,
            temperature=config.mistral.temperature,
            max_tokens=config.mistral.max_tokens,
            timeout=config.mistral.timeout,
            max_retries=config.mistral.max_retries
        )
    elif provider_name == 'mock':
        return MockLLMProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
