"""
editor.py
─────────────────────────────────────────────────────────────────────────────
Phase 3 & 4 — Smart Video Reframing + Dynamic AI Editing

Responsibilities:
  • Extract clip from source video (ffmpeg)
  • Face-detection & motion-tracking via MediaPipe (active-speaker follow)
  • Smart 9:16 reframe with smooth pan, or stacked split-screen for 2 speakers
  • Kinetic Alex Hormozi-style word-by-word captions (PIL → frame overlay)
  • B-roll overlay from Pexels or AI-generated stills
  • Retention progress bar + Subscribe CTA in final 3 seconds
  • Subtle SFX injection (whoosh on cuts, pop on key words)
  • Final H.264/AAC encode to output directory
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import httpx
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from analyzer import ViralClip, WordTimestamp
from config.settings import settings
from downloader import VideoAsset


mp_face = mp.solutions.face_detection


# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class FaceBox:
    x: int; y: int; w: int; h: int   # pixel coords in the original frame
    confidence: float

@dataclass
class CropWindow:
    """Defines the 9:16 crop rectangle within the original 16:9 frame."""
    x: int         # left edge in source pixels
    y: int         # top edge in source pixels
    width: int     # crop width  (source pixels)
    height: int    # crop height (source pixels)


# ─── B-Roll fetcher ───────────────────────────────────────────────────────────

class BRollFetcher:
    """Downloads B-roll clips or stills from Pexels."""

    BASE = "https://api.pexels.com/videos/search"

    def __init__(self, api_key: str, cache_dir: Path):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, keyword: str, duration_hint: float = 5.0) -> Optional[Path]:
        if not self.api_key:
            logger.warning("[broll] No Pexels key; skipping B-roll")
            return None

        safe_kw = re.sub(r"[^\w]", "_", keyword.lower())
        cache_path = self.cache_dir / f"broll_{safe_kw}.mp4"
        if cache_path.exists():
            return cache_path

        try:
            headers = {"Authorization": self.api_key}
            resp = httpx.get(
                self.BASE,
                params={"query": keyword, "per_page": 5, "size": "medium"},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            videos = resp.json().get("videos", [])
            if not videos:
                return None

            # Pick a video close in duration to our desired window
            chosen = min(
                videos,
                key=lambda v: abs(v.get("duration", 99) - duration_hint),
            )
            files = chosen.get("video_files", [])
            # prefer 1080p mp4
            hd_files = [
                f for f in files
                if f.get("quality") in ("hd", "sd") and "mp4" in f.get("file_type", "")
            ]
            if not hd_files:
                return None

            video_url = hd_files[0]["link"]
            with httpx.stream("GET", video_url, timeout=60) as stream:
                with open(cache_path, "wb") as fh:
                    for chunk in stream.iter_bytes(chunk_size=65536):
                        fh.write(chunk)
            logger.debug(f"[broll] Downloaded: {cache_path.name}")
            return cache_path

        except Exception as exc:
            logger.warning(f"[broll] Fetch failed for {keyword!r}: {exc}")
            return None


# ─── Caption Renderer ─────────────────────────────────────────────────────────

# Emoji shortcuts for common punchy words
EMOJI_MAP = {
    "money": "💰", "win": "🏆", "fire": "🔥", "love": "❤️",
    "crazy": "🤯", "secret": "🤫", "alert": "🚨", "fast": "⚡",
    "grow": "📈", "goal": "🎯", "yes": "✅", "no": "❌",
}
PUNCHY_WORDS = {
    "never", "always", "secret", "revealed", "shocking", "insane",
    "literally", "actually", "honestly", "crazy", "incredible",
    "stop", "start", "now", "immediately", "first", "last", "biggest",
    "million", "billion", "free", "viral", "hack", "truth", "lie",
}

def _is_number(word: str) -> bool:
    return bool(re.match(r"^\$?[\d,\.]+[kmb%]?$", word.lower().strip(".,!?")))

def _emoji_for(word: str) -> str:
    clean = word.lower().strip(".,!?\"'")
    return EMOJI_MAP.get(clean, "")


class CaptionRenderer:
    """
    Renders kinetic word-by-word captions onto a PIL frame.
    Key words scale up + change colour; numbers go green.
    """

    def __init__(self):
        self.font_size = settings.caption_font_size
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}
        self._load_fonts()

    def _load_fonts(self):
        # Try to load Impact or fall back to a default
        font_candidates = [
            "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Impact.ttf",   # macOS
        ]
        self._font_path: Optional[str] = None
        for p in font_candidates:
            if os.path.exists(p):
                self._font_path = p
                break
        if self._font_path is None:
            logger.warning("[captions] Impact font not found; using default")

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        if size not in self._font_cache:
            if self._font_path:
                self._font_cache[size] = ImageFont.truetype(self._font_path, size)
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def render_frame(
        self,
        frame_rgb: np.ndarray,
        active_words: list[WordTimestamp],
        current_time: float,
    ) -> np.ndarray:
        """
        Overlay the currently-active caption word(s) onto frame_rgb (H×W×3).
        Returns modified frame.
        """
        if not active_words:
            return frame_rgb

        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img)

        W, H = img.size
        y_base = int(H * 0.82)   # caption sits at ~82% height

        # Collect words to render (current ± 1 word context)
        current_idx = None
        for i, w in enumerate(active_words):
            if w.start <= current_time <= w.end:
                current_idx = i
                break

        if current_idx is None:
            return frame_rgb

        # Render a window of words centred on the current word
        window = active_words[max(0, current_idx - 2): current_idx + 3]
        total_width = 0
        render_items = []

        for w in window:
            is_current = (w.start <= current_time <= w.end)
            word_clean = w.word.strip(".,!?\"'")
            is_punchy = word_clean.lower() in PUNCHY_WORDS
            is_num = _is_number(word_clean)
            emoji = _emoji_for(word_clean)

            if is_num:
                color = settings.caption_number_color
            elif is_punchy or is_current:
                color = settings.caption_highlight_color
            else:
                color = settings.caption_base_color

            scale = 1.35 if (is_current or is_punchy) else 1.0
            fsize = int(self.font_size * scale)
            font = self._get_font(fsize)

            display = w.word.upper()
            if emoji:
                display = f"{display} {emoji}"

            bbox = draw.textbbox((0, 0), display, font=font)
            tw = bbox[2] - bbox[0]
            render_items.append((display, font, color, tw, fsize, is_current))
            total_width += tw + 14   # 14px letter-spacing

        # Centre the word strip
        x = (W - total_width) // 2
        for display, font, color, tw, fsize, is_current in render_items:
            y = y_base - fsize
            # Stroke / shadow
            for dx in (-settings.caption_stroke_width, settings.caption_stroke_width):
                for dy in (-settings.caption_stroke_width, settings.caption_stroke_width):
                    draw.text(
                        (x + dx, y + dy), display,
                        font=font, fill=settings.caption_stroke_color
                    )
            draw.text((x, y), display, font=font, fill=color)
            x += tw + 14

        return np.array(img)


# ─── Face Tracker ─────────────────────────────────────────────────────────────

class FaceTracker:
    """
    Uses MediaPipe Face Detection to find the dominant speaker per frame
    and maintain a smoothed crop window for 9:16 output.
    """

    def __init__(self, src_w: int, src_h: int, out_w: int, out_h: int):
        self.src_w = src_w
        self.src_h = src_h
        self.out_w = out_w
        self.out_h = out_h
        # The crop width in source space that maps to out_w×out_h ratio
        self.crop_h = src_h
        self.crop_w = int(src_h * (out_w / out_h))
        self._detector = mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.45
        )
        # Smoothing: exponential moving average on crop-centre x
        self._smooth_cx: float = src_w / 2
        self._alpha = 0.12   # lower = smoother but laggier

    def compute_crop(self, frame_bgr: np.ndarray) -> CropWindow:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.process(frame_rgb)

        if result.detections:
            # Pick the largest (most prominent) face
            best = max(
                result.detections,
                key=lambda d: (
                    d.location_data.relative_bounding_box.width *
                    d.location_data.relative_bounding_box.height
                )
            )
            bb = best.location_data.relative_bounding_box
            face_cx = (bb.xmin + bb.width / 2) * self.src_w
            self._smooth_cx = (
                self._alpha * face_cx + (1 - self._alpha) * self._smooth_cx
            )

        # Clamp to valid range
        cx = int(self._smooth_cx)
        x = max(0, min(cx - self.crop_w // 2, self.src_w - self.crop_w))
        return CropWindow(x=x, y=0, width=self.crop_w, height=self.crop_h)

    def close(self):
        self._detector.close()


# ─── Progress Bar & CTA Overlay ───────────────────────────────────────────────

def draw_progress_bar(
    frame_rgb: np.ndarray,
    progress: float,           # 0.0 → 1.0
    clip_duration: float,
    current_time: float,
    show_cta: bool = False,
) -> np.ndarray:
    """
    Draws a thin retention progress bar at the bottom of the frame.
    During the last 3 seconds, also draws a Subscribe CTA banner.
    """
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # Progress bar (6 px tall)
    bar_h = 6
    bar_y = H - bar_h - 2
    draw.rectangle([0, bar_y, W, H], fill=(0, 0, 0, 160))
    fill_w = int(W * progress)
    draw.rectangle([0, bar_y, fill_w, H], fill=(255, 80, 80, 220))

    if show_cta:
        # Subscribe banner
        cta_h = 70
        cta_y = H - bar_h - cta_h - 10
        draw.rectangle(
            [W // 4, cta_y, 3 * W // 4, cta_y + cta_h],
            fill=(255, 0, 0, 200), outline=(255, 255, 255, 255), width=2
        )
        try:
            font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
            font = ImageFont.truetype(font_path, 32) if os.path.exists(font_path) else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        draw.text(
            (W // 2, cta_y + cta_h // 2),
            "🔔 SUBSCRIBE",
            font=font,
            fill=(255, 255, 255, 255),
            anchor="mm",
        )

    return np.array(img)


# ─── Master Editor ────────────────────────────────────────────────────────────

class VideoEditor:
    """
    Produces a polished 9:16 vertical short for each ViralClip.
    """

    def __init__(self, asset: VideoAsset, words: list[WordTimestamp]):
        self.asset = asset
        self.words = words
        self.broll = BRollFetcher(
            settings.pexels_api_key,
            settings.output_dir / "assets" / "broll"
        )
        self.caption_renderer = CaptionRenderer()
        self.out_w, self.out_h = settings.output_resolution

    # ── Public entry ──────────────────────────────────────────────────────

    def render_clip(self, clip: ViralClip, clip_index: int) -> Path:
        """Full render pipeline; returns path to finished MP4."""
        out_path = (
            settings.output_dir / "clips"
            / f"{self.asset.video_id}_clip{clip_index:02d}.mp4"
        )
        if out_path.exists():
            logger.info(f"[editor] Already rendered: {out_path.name}")
            return out_path

        logger.info(
            f"[editor] Rendering clip {clip_index}: "
            f"{clip.start_seconds:.1f}s → {clip.end_seconds:.1f}s "
            f"(score {clip.virality_score}/10)"
        )

        # 1. Extract raw clip
        raw_clip = self._extract_raw_clip(clip, clip_index)

        # 2. Fetch B-roll assets
        broll_paths: dict[str, Optional[Path]] = {}
        for overlay in clip.visual_overlays:
            p = self.broll.fetch(overlay.keyword, overlay.duration_seconds)
            broll_paths[overlay.keyword] = p

        # 3. Frame-by-frame processing (face track + captions + progress bar)
        processed_clip = self._process_frames(
            raw_clip, clip, clip_index, broll_paths
        )

        # 4. Final encode: mux audio + video
        self._final_encode(raw_clip, processed_clip, out_path)

        raw_clip.unlink(missing_ok=True)
        processed_clip.unlink(missing_ok=True)

        logger.success(f"[editor] Done: {out_path.name}")
        return out_path

    # ── Step 1: Raw clip extraction ───────────────────────────────────────

    def _extract_raw_clip(self, clip: ViralClip, idx: int) -> Path:
        tmp = settings.output_dir / "clips" / f"_raw_{self.asset.video_id}_{idx}.mp4"
        duration = clip.end_seconds - clip.start_seconds
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip.start_seconds),
            "-i", str(self.asset.video_path),
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(tmp),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp

    # ── Step 2: Frame-level processing ───────────────────────────────────

    def _process_frames(
        self,
        raw_clip: Path,
        clip: ViralClip,
        clip_index: int,
        broll_paths: dict[str, Optional[Path]],
    ) -> Path:
        cap = cv2.VideoCapture(str(raw_clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or settings.output_fps
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_duration = clip.end_seconds - clip.start_seconds

        tracker = FaceTracker(src_w, src_h, self.out_w, self.out_h)

        # Temp video output (no audio)
        tmp_video = (
            settings.output_dir / "clips"
            / f"_proc_{self.asset.video_id}_{clip_index}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(tmp_video), fourcc, fps, (self.out_w, self.out_h)
        )

        # Filter words to this clip's time range
        clip_words = [
            WordTimestamp(
                word=w.word,
                start=w.start - clip.start_seconds,
                end=w.end - clip.start_seconds,
                confidence=w.confidence,
            )
            for w in self.words
            if w.start >= clip.start_seconds and w.end <= clip.end_seconds
        ]

        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            t = frame_idx / fps          # relative time within clip
            progress = t / clip_duration if clip_duration > 0 else 0
            show_cta = (clip_duration - t) <= 3.0

            # ── Face-aware crop ──────────────────────────────────────────
            crop = tracker.compute_crop(frame_bgr)
            cropped = frame_bgr[
                crop.y : crop.y + crop.height,
                crop.x : crop.x + crop.width,
            ]
            resized = cv2.resize(cropped, (self.out_w, self.out_h))
            frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # ── B-Roll overlay (if in window) ────────────────────────────
            frame_rgb = self._maybe_overlay_broll(
                frame_rgb, t, clip, broll_paths, fps
            )

            # ── Kinetic captions ─────────────────────────────────────────
            frame_rgb = self.caption_renderer.render_frame(
                frame_rgb, clip_words, t
            )

            # ── Progress bar + CTA ───────────────────────────────────────
            frame_rgb = draw_progress_bar(
                frame_rgb, progress, clip_duration, t, show_cta
            )

            # Write back
            out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)
            frame_idx += 1

        cap.release()
        writer.release()
        tracker.close()
        return tmp_video

    # ── B-Roll overlay ────────────────────────────────────────────────────

    def _maybe_overlay_broll(
        self,
        frame_rgb: np.ndarray,
        t: float,
        clip: ViralClip,
        broll_paths: dict[str, Optional[Path]],
        fps: float,
    ) -> np.ndarray:
        """Pip-style B-roll inset in top-left corner."""
        for overlay in clip.visual_overlays:
            win_start = overlay.timestamp_seconds - clip.start_seconds
            win_end = win_start + overlay.duration_seconds
            if not (win_start <= t < win_end):
                continue

            broll_file = broll_paths.get(overlay.keyword)
            if broll_file is None or not broll_file.exists():
                continue

            try:
                local_t = t - win_start
                cap = cv2.VideoCapture(str(broll_file))
                b_fps = cap.get(cv2.CAP_PROP_FPS) or 25
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(local_t * b_fps))
                ok, b_frame = cap.read()
                cap.release()
                if not ok:
                    continue

                # Inset: top-right corner, 35% width
                inset_w = int(self.out_w * 0.35)
                inset_h = int(inset_w * 9 / 16)
                b_resized = cv2.resize(
                    cv2.cvtColor(b_frame, cv2.COLOR_BGR2RGB),
                    (inset_w, inset_h)
                )
                pad = 20
                frame_rgb[pad: pad + inset_h,
                           self.out_w - inset_w - pad: self.out_w - pad] = b_resized
            except Exception as exc:
                logger.debug(f"[editor] B-roll overlay error: {exc}")

        return frame_rgb

    # ── Final encode ──────────────────────────────────────────────────────

    def _final_encode(
        self,
        raw_clip: Path,
        processed_video: Path,
        out_path: Path,
    ) -> None:
        """Mux processed video frames with original audio; apply loudness norm."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(processed_video),
            "-i", str(raw_clip),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "slow",
            "-crf", str(settings.output_crf),
            "-profile:v", "high", "-level", "4.1",
            "-c:a", "aac", "-b:a", "192k",
            "-ar", "44100",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-movflags", "+faststart",
            "-shortest",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
