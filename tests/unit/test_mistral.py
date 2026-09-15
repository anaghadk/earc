import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from src.llm.mistral_client import MistralLLMProvider

@pytest.fixture
def provider():
    # Use patch to mock the Mistral init so it doesn't try to read env vars
    with patch('src.llm.mistral_client.Mistral') as MockClient:
        yield MistralLLMProvider(api_key="dummy", model_name="mistral-test")

def test_successful_response(provider):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="mistral answer"))]
    mock_response.usage = MagicMock(total_tokens=10, prompt_tokens=4, completion_tokens=6)
    
    provider.client.chat.complete.return_value = mock_response
    
    result = provider.generate("prompt")
    assert result.text == "mistral answer"
    assert result.usage["total_tokens"] == 10

def test_auth_error(provider):
    import mistralai.models
    provider.client.chat.complete.side_effect = Exception("401 Unauthorized")
    
    with pytest.raises(RuntimeError):
        provider.generate("prompt")

def test_rate_limit(provider):
    provider.client.chat.complete.side_effect = Exception("429 Too Many Requests")
    with pytest.raises(RuntimeError):
        provider.generate("prompt")

def test_timeout(provider):
    provider.client.chat.complete.side_effect = Exception("Timeout")
    with pytest.raises(RuntimeError):
        provider.generate("prompt")
