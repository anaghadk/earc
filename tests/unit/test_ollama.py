import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from src.llm.ollama_client import OllamaLLMProvider

@pytest.fixture
def provider():
    return OllamaLLMProvider(model_name="test-model", base_url="http://localhost:11434")

@patch('httpx.post')
def test_successful_response(mock_post, provider):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "test answer", "eval_count": 5}
    mock_post.return_value = mock_response

    result = provider.generate("test prompt")
    assert result.text == "test answer"
    assert result.usage["total_tokens"] == 5

@patch('httpx.post')
def test_connection_error(mock_post, provider):
    import httpx
    mock_post.side_effect = httpx.RequestError("Connection failed")
    
    with pytest.raises(RuntimeError) as excinfo:
        provider.generate("test prompt")
    assert "Ollama generation failed" in str(excinfo.value)

@patch('httpx.post')
def test_timeout(mock_post, provider):
    import httpx
    mock_post.side_effect = httpx.TimeoutException("Timeout")
    
    with pytest.raises(RuntimeError):
        provider.generate("test prompt")

@patch('httpx.post')
def test_malformed_response(mock_post, provider):
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Missing 'response' key
    mock_response.json.return_value = {"something_else": "test answer"}
    mock_post.return_value = mock_response

    with pytest.raises(RuntimeError):
        provider.generate("test prompt")
