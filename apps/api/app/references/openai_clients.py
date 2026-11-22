from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import httpx
from openai import OpenAI

from app.references.files import write_reference_raw_stt, write_reference_script


def compress_audio_for_transcription(audio_path: Path) -> Path | None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None
    temp_dir = audio_path.parent
    temp_mp3 = temp_dir / f".compressed_{audio_path.stem}.mp3"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(audio_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-b:a",
        "48k",
        str(temp_mp3),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, check=False)
        if res.returncode == 0 and temp_mp3.exists() and temp_mp3.stat().st_size > 0:
            return temp_mp3
    except Exception:
        pass
    return None


class TranscriptionClient(Protocol):
    def transcribe(self, audio_path: Path) -> dict:
        ...


class OpenAITranscriptionClient:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OpenAI API key is missing")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def transcribe(self, audio_path: Path) -> dict:
        transcribe_path = audio_path
        cleanup_temp = False

        if audio_path.stat().st_size > 24 * 1024 * 1024:
            compressed = compress_audio_for_transcription(audio_path)
            if compressed and compressed.exists():
                transcribe_path = compressed
                cleanup_temp = True
            elif audio_path.stat().st_size > 25 * 1024 * 1024:
                raise ValueError(
                    f"오디오 파일 크기({audio_path.stat().st_size // (1024*1024)}MB)가 Whisper API 한도(25MB)를 초과했습니다. ffmpeg를 사용하거나 더 짧은 영상을 선택해주세요."
                )

        try:
            with transcribe_path.open("rb") as audio_file:
                result = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return dict(result)
        finally:
            if cleanup_temp and transcribe_path.exists():
                try:
                    transcribe_path.unlink()
                except OSError:
                    pass


class GeminiTranscriptionClient:
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        if not api_key:
            raise ValueError("Gemini API key is missing")
        self.api_key = api_key
        model_name = (model or "").strip()
        if not model_name or model_name in ("gemini-2.5-flash", "gemini-1.5-flash", "gemini-flash"):
            self.model = "gemini-3.8-flash"
        elif model_name in ("gemini-2.5-pro", "gemini-1.5-pro", "gemini-pro"):
            self.model = "gemini-3.1-pro-preview"
        else:
            self.model = model_name

    def transcribe(self, audio_path: Path) -> dict:
        transcribe_path = audio_path
        cleanup_temp = False

        if audio_path.stat().st_size > 18 * 1024 * 1024:
            compressed = compress_audio_for_transcription(audio_path)
            if compressed and compressed.exists():
                transcribe_path = compressed
                cleanup_temp = True

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            audio_bytes = transcribe_path.read_bytes()
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            suffix = transcribe_path.suffix.lower()
            mime = "audio/mp3" if suffix == ".mp3" else "audio/wav" if suffix == ".wav" else "audio/mpeg"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Transcribe the following audio exactly in Korean without any extra explanation. Only return the verbatim transcribed speech."},
                            {"inlineData": {"mimeType": mime, "data": b64_audio}},
                        ]
                    }
                ]
            }
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    raise RuntimeError(f"Gemini transcription failed ({resp.status_code}): {resp.text}")
                data = resp.json()

            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"Gemini transcription returned no candidates: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = (parts[0].get("text", "") if parts else "").strip()
            return {"text": text, "segments": [{"text": text, "start": 0.0, "end": 0.0}]}
        finally:
            if cleanup_temp and transcribe_path.exists():
                try:
                    transcribe_path.unlink()
                except OSError:
                    pass


class AnalysisClient(Protocol):
    def analyze(self, prompt: str) -> dict:
        ...


class OpenAIAnalysisClient:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OpenAI API key is missing")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze(self, prompt: str) -> dict:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )
        return json.loads(response.output_text)


class GeminiAnalysisClient:
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        if not api_key:
            raise ValueError("Gemini API key is missing")
        self.api_key = api_key
        model_name = (model or "").strip()
        if not model_name or model_name in ("gemini-2.5-flash", "gemini-1.5-flash", "gemini-flash"):
            self.model = "gemini-3.8-flash"
        elif model_name in ("gemini-2.5-pro", "gemini-1.5-pro", "gemini-pro"):
            self.model = "gemini-3.1-pro-preview"
        else:
            self.model = model_name

    def analyze(self, prompt: str) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API request failed ({resp.status_code}): {resp.text}")
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError(f"Gemini returned empty parts: {data}")

        text = parts[0].get("text", "").strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())


def write_reference_stt_outputs(project_dir: Path, raw_stt: dict) -> str:
    text = str(raw_stt.get("text", "")).strip()
    write_reference_raw_stt(project_dir, raw_stt)
    write_reference_script(project_dir, f"{text}\n")
    return text
