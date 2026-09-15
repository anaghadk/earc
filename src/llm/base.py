"""
Abstract base classes for LLM providers and token counting.

The EARC framework supports multiple generation backends
(Ollama, Mistral API) via dependency injection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.schemas import GenerationResult


class TokenCounter(ABC):
    """
    Abstract token counter.

    Token counting must match the generation backend being evaluated
    (paper §Stage 7). Do NOT approximate with len(text.split()).
    """

    @abstractmethod
    def count(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: Input text string.

        Returns:
            Number of tokens.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tokenizer/model used for counting."""
        ...


class LLMProvider(ABC):
    """
    Abstract LLM provider interface.

    All LLM backends (Ollama, Mistral) implement this interface.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The full prompt string.
            **kwargs: Provider-specific generation parameters.

        Returns:
            GenerationResult with text, usage, latency, and error info.
        """
        ...

    @abstractmethod
    def check_availability(self) -> bool:
        """
        Check whether the provider is reachable and the model is available.

        Returns:
            True if the provider is ready, False otherwise.
        """
        ...

    @abstractmethod
    def get_token_counter(self) -> TokenCounter:
        """
        Return a TokenCounter appropriate for this provider's model.

        The token counter is used during budget selection (§Stage 7)
        to ensure compressed contexts stay within the token limit.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (e.g., 'ollama', 'mistral')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier (e.g., 'llama3.2', 'mistral-small-latest')."""
        ...
