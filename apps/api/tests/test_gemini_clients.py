from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.references.openai_clients import GeminiAnalysisClient, GeminiTranscriptionClient


def test_gemini_analysis_client_requires_api_key():
    with pytest.raises(ValueError, match="Gemini API key is missing"):
        GeminiAnalysisClient(api_key="")


@patch("httpx.Client.post")
def test_gemini_analysis_client_parses_json_response(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": '{"result": "success", "title": "Gemini Test"}'}
                    ]
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    client = GeminiAnalysisClient(api_key="fake-gemini-key")
    res = client.analyze("test prompt")

    assert res == {"result": "success", "title": "Gemini Test"}
    mock_post.assert_called_once()
    assert "fake-gemini-key" in mock_post.call_args[0][0]


@patch("httpx.Client.post")
def test_gemini_analysis_client_strips_markdown_codeblock(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "```json\n{\"analysis\": \"complete\"}\n```"}
                    ]
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    client = GeminiAnalysisClient(api_key="fake-gemini-key")
    res = client.analyze("test prompt")

    assert res == {"analysis": "complete"}


def test_gemini_transcription_client_requires_api_key():
    with pytest.raises(ValueError, match="Gemini API key is missing"):
        GeminiTranscriptionClient(api_key="")


@patch("httpx.Client.post")
def test_gemini_transcription_client_transcribes(mock_post, tmp_path: Path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"dummy audio data")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "안녕하세요 제미나이 전사 테스트입니다."}
                    ]
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    client = GeminiTranscriptionClient(api_key="fake-gemini-key")
    res = client.transcribe(audio_file)

    assert res["text"] == "안녕하세요 제미나이 전사 테스트입니다."
    assert len(res["segments"]) == 1
