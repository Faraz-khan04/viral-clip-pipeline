"""
analyzer.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Multimodal Virality & Hook Analysis

Responsibilities:
  • Transcribe audio with AssemblyAI (word-level timestamps)
  • Compute per-second audio energy / loudness envelope
  • Merge retention heatmap + energy + transcript into a feature vector
  • Call Claude (or GPT-4o) with strict Pydantic-validated structured output
    to identify the top viral clip windows
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import anthropic
import assemblyai as aai
import librosa
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import settings
from downloader import RetentionPoint, VideoAsset


# ─── Pydantic Output Schemas ──────────────────────────────────────────────────

class VisualOverlay(BaseModel):
    keyword: str = Field(description="Key term to search for B-roll")
    timestamp_seconds: float = Field(description="When to show the B-roll")
    duration_seconds: float = Field(default=3.0)
    source: str = Field(default="pexels", description="'pexels' or 'ai_generated'")


class ViralClip(BaseModel):
    start_seconds: float = Field(description="Clip start in source video (seconds)")
    end_seconds: float = Field(description="Clip end in source video (seconds)")
    virality_score: float = Field(
        ge=0, le=10,
        description="0–10 composite score blending hook strength, emotion & retention"
    )
    hook_sentence: str = Field(description="The first compelling sentence of the clip")
    vertical_title: str = Field(
        max_length=100,
        description="Short, punchy title optimised for Shorts / Reels"
    )
    seo_caption: str = Field(
        max_length=2200,
        description="SEO-rich caption with hashtags for all platforms"
    )
    hashtags: list[str] = Field(default_factory=list, max_length=30)
    visual_overlays: list[VisualOverlay] = Field(default_factory=list)
    first_pinned_comment: str = Field(
        max_length=500,
        description="Automated first comment CTA with link back to full video"
    )

    @field_validator("end_seconds")
    @classmethod
    def clip_must_be_positive(cls, v, info):
        start = info.data.get("start_seconds", 0)
        if v <= start:
            raise ValueError("end_seconds must be > start_seconds")
        return v

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


class AnalysisResult(BaseModel):
    video_id: str
    transcript: str
    clips: list[ViralClip]
    total_duration: float


# ─── Word-level transcript entry ──────────────────────────────────────────────

class WordTimestamp(BaseModel):
    word: str
    start: float   # seconds
    end: float     # seconds
    confidence: float = 1.0


# ─── Transcriber ─────────────────────────────────────────────────────────────

class AssemblyAITranscriber:
    """Returns full text + word-level timestamps via AssemblyAI."""

    def __init__(self, api_key: str):
        aai.settings.api_key = api_key

    def transcribe(self, audio_path: Path) -> tuple[str, list[WordTimestamp]]:
        logger.info(f"[transcriber] Submitting: {audio_path.name}")
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            language_detection=True,
        )
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(str(audio_path))

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(
                f"AssemblyAI transcription error: {transcript.error}"
            )

        full_text = transcript.text or ""
        words: list[WordTimestamp] = []
        for w in transcript.words or []:
            words.append(WordTimestamp(
                word=w.text,
                start=w.start / 1000.0,   # ms → seconds
                end=w.end / 1000.0,
                confidence=w.confidence,
            ))

        logger.success(
            f"[transcriber] Done. {len(words)} words, "
            f"{len(full_text)} chars"
        )
        return full_text, words


# ─── Audio Energy Analyser ────────────────────────────────────────────────────

class AudioEnergyAnalyser:
    """
    Computes per-second RMS energy envelope & marks loud moments
    that correlate with high-emotion events (laughter, applause, debate).
    """

    def __init__(self, frame_duration: float = 1.0):
        self.frame_duration = frame_duration   # seconds per energy bucket

    def analyse(self, audio_path: Path) -> list[dict]:
        """
        Returns list of {start, end, rms_db, is_spike} dicts,
        one per second of the audio.
        """
        logger.info(f"[audio] Loading: {audio_path.name}")
        y, sr = librosa.load(str(audio_path), sr=16_000, mono=True)
        hop = int(sr * self.frame_duration)

        # RMS per frame
        rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max)

        # Spike detection: > 1 std above mean
        mean_db = float(np.mean(rms_db))
        std_db = float(np.std(rms_db))
        threshold = mean_db + std_db

        frames = []
        for i, db in enumerate(rms_db):
            start = i * self.frame_duration
            frames.append({
                "start": start,
                "end": start + self.frame_duration,
                "rms_db": round(float(db), 2),
                "is_spike": bool(float(db) > threshold),
            })

        spikes = sum(1 for f in frames if f["is_spike"])
        logger.info(f"[audio] {len(frames)} frames, {spikes} energy spikes")
        return frames


# ─── LLM Virality Analyser ───────────────────────────────────────────────────

class LLMViralityAnalyser:
    """
    Calls Claude with a rich context payload and returns structured ViralClip
    objects via Pydantic validation.
    """

    SYSTEM_PROMPT = """
You are an elite viral video strategist with deep expertise in social media
growth, audience psychology, and short-form content (YouTube Shorts, TikTok,
Instagram Reels).

You will be given a timestamped transcript, audio energy spikes, and
YouTube Most-Replayed heatmap data for a long-form video.

Your job:
1. Identify the top viral clip windows (60–90 seconds ideal; ≤180 seconds max).
2. Every clip MUST start with a strong hook (question, bold claim, shocking
   stat, or emotional peak) that grabs attention within the first 3 seconds.
3. Prioritise segments where: retention heatmap intensity > 0.6, energy spikes
   cluster, and the transcript contains high-emotion language.
4. For each clip, generate an optimised vertical title, SEO caption, hashtags,
   suggested B-roll keyword overlays, and a first pinned comment CTA.

Return ONLY a JSON object matching the schema provided. No prose. No markdown.
""".strip()

    def __init__(self):
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )

    @retry(
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.APIStatusError)
        ),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda rs: logger.warning(
            f"[llm] Rate limited; retry {rs.attempt_number}…"
        ),
    )
    def analyse(
        self,
        asset: VideoAsset,
        transcript: str,
        words: list[WordTimestamp],
        energy_frames: list[dict],
    ) -> list[ViralClip]:

        payload = self._build_payload(asset, transcript, words, energy_frames)
        schema = AnalysisResult.model_json_schema()

        logger.info(
            f"[llm] Sending {len(payload['transcript_chunks'])} "
            f"transcript chunks to {settings.llm_model}"
        )

        user_message = (
            f"VIDEO METADATA:\n"
            f"Title: {asset.title}\n"
            f"Duration: {asset.duration_seconds:.0f}s\n"
            f"Views: {asset.view_count:,}\n\n"
            f"RETENTION HEATMAP (top spikes):\n"
            f"{json.dumps(payload['top_retention_spikes'], indent=2)}\n\n"
            f"AUDIO ENERGY SPIKES:\n"
            f"{json.dumps(payload['energy_spikes'], indent=2)}\n\n"
            f"FULL TRANSCRIPT WITH TIMESTAMPS (every 30-second chunk):\n"
            f"{json.dumps(payload['transcript_chunks'], indent=2)}\n\n"
            f"Required output JSON schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Return ONLY valid JSON. No markdown fences."
        )

        response = self._client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        # Strip accidental markdown fences
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()

        parsed = json.loads(raw)
        # Inject video_id and transcript if missing
        parsed.setdefault("video_id", asset.video_id)
        parsed.setdefault("transcript", transcript[:500])
        parsed.setdefault("total_duration", asset.duration_seconds)

        result = AnalysisResult.model_validate(parsed)

        # Filter by minimum virality score
        clips = [
            c for c in result.clips
            if c.virality_score >= settings.min_clip_virality_score
        ]
        # Sort best-first, cap count
        clips.sort(key=lambda c: c.virality_score, reverse=True)
        clips = clips[: settings.max_clips_per_video]

        logger.success(
            f"[llm] {len(clips)} clips above score threshold "
            f"({settings.min_clip_virality_score})"
        )
        return clips

    # ── Payload builder ───────────────────────────────────────────────────

    @staticmethod
    def _build_payload(
        asset: VideoAsset,
        transcript: str,
        words: list[WordTimestamp],
        energy_frames: list[dict],
    ) -> dict:
        # Top 20 retention spikes
        sorted_retention = sorted(
            asset.retention_graph,
            key=lambda r: r.intensity,
            reverse=True,
        )[:20]
        top_retention = [
            {
                "start_sec": r.start_millis / 1000,
                "intensity": r.intensity,
            }
            for r in sorted_retention
        ]

        # Energy spike windows
        energy_spikes = [
            {"start": f["start"], "end": f["end"], "rms_db": f["rms_db"]}
            for f in energy_frames if f["is_spike"]
        ]

        # Group words into 30-second chunks for context
        chunks = []
        chunk_size = 30.0
        if words:
            duration = words[-1].end
            n_chunks = math.ceil(duration / chunk_size)
            for i in range(n_chunks):
                t_start = i * chunk_size
                t_end = t_start + chunk_size
                chunk_words = [
                    w for w in words if w.start >= t_start and w.start < t_end
                ]
                chunks.append({
                    "time_range": f"{t_start:.0f}s–{t_end:.0f}s",
                    "text": " ".join(w.word for w in chunk_words),
                })
        else:
            # Fallback: split plain transcript
            words_flat = transcript.split()
            step = max(1, len(words_flat) // 20)
            for i in range(0, len(words_flat), step):
                chunks.append({
                    "time_range": "approx",
                    "text": " ".join(words_flat[i : i + step]),
                })

        return {
            "top_retention_spikes": top_retention,
            "energy_spikes": energy_spikes[:50],  # cap size
            "transcript_chunks": chunks,
        }


# ─── Master Analyser (facade) ────────────────────────────────────────────────

class VideoAnalyser:
    """Orchestrates transcription, audio analysis, and LLM virality scoring."""

    def __init__(self):
        self.transcriber = AssemblyAITranscriber(settings.assemblyai_api_key)
        self.audio_analyser = AudioEnergyAnalyser()
        self.llm_analyser = LLMViralityAnalyser()

    def run(self, asset: VideoAsset) -> tuple[list[ViralClip], list[WordTimestamp]]:
        """
        Returns (clips, word_timestamps) where clips are sorted best-first.
        """
        # 1. Transcribe
        transcript, words = self.transcriber.transcribe(asset.audio_path)

        # 2. Audio energy
        energy_frames = self.audio_analyser.analyse(asset.audio_path)

        # 3. LLM scoring
        clips = self.llm_analyser.analyse(asset, transcript, words, energy_frames)

        return clips, words
