import base64
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.production.service import _run_gemini_image_generate


def test_gemini_image_generate_requires_api_key(tmp_path: Path):
    with pytest.raises(ValueError, match="Gemini API key is required"):
        _run_gemini_image_generate(
            prompt="A cute cat",
            output_path=tmp_path / "out.png",
            api_key="",
        )


@patch("httpx.Client.post")
def test_gemini_image_generate_success(mock_post, tmp_path: Path):
    dummy_png = b"\x89PNG\r\n\x1a\n" + b"fake png content"
    b64_dummy = base64.b64encode(dummy_png).decode("utf-8")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "predictions": [
            {
                "bytesBase64Encoded": b64_dummy,
                "mimeType": "image/png"
            }
        ]
    }
    mock_post.return_value = mock_response

    output_path = tmp_path / "scene_01_start.png"
    result = _run_gemini_image_generate(
        prompt="A 3D claymation scene",
        output_path=output_path,
        api_key="fake-gemini-key",
        aspect_ratio="9:16",
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.read_bytes() == dummy_png
    mock_post.assert_called_once()
    assert "fake-gemini-key" in mock_post.call_args[0][0]
