from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Protocol

import httpx
from openai import OpenAI

from app.references.files import write_reference_raw_stt, write_reference_script


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
        with audio_path.open("rb") as audio_file:
            result = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return dict(result)


class GeminiTranscriptionClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("Gemini API key is missing")
        self.api_key = api_key
        self.model = model or "gemini-2.5-flash"

    def transcribe(self, audio_path: Path) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        audio_bytes = audio_path.read_bytes()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        mime = "audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/mp3"
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
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("Gemini API key is missing")
        self.api_key = api_key
        self.model = model or "gemini-2.5-flash"
        if not self.model.startswith("gemini-"):
            self.model = "gemini-2.5-flash"

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
