"""
downloader.py
─────────────────────────────────────────────────────────────────────────────
Phase 1 — Intelligent Ingestion & Metadata Scraping

Responsibilities:
  • Download long-form video + audio track via yt-dlp
  • Fetch native / auto-generated subtitles (VTT → plain text)
  • Scrape the YouTube "Most Replayed" heatmap data from the page's
    ytInitialData blob (no paid API required)
  • Return a fully-populated VideoAsset dataclass to the pipeline
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

import httpx
import yt_dlp
from loguru import logger


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class RetentionPoint:
    """Single point on the YouTube Most-Replayed heatmap."""
    start_millis: int
    duration_millis: int
    intensity: float          # 0.0 – 1.0, normalised

@dataclass
class VideoAsset:
    video_id: str
    title: str
    description: str
    duration_seconds: float
    video_path: Path
    audio_path: Path
    subtitle_path: Optional[Path]
    thumbnail_url: str
    channel_name: str
    view_count: int
    retention_graph: list[RetentionPoint] = field(default_factory=list)
    raw_metadata: dict = field(default_factory=dict)


# ─── Downloader ───────────────────────────────────────────────────────────────

class VideoDownloader:
    """Downloads video, audio, subtitles and retention data."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────

    def download(self, url: str) -> VideoAsset:
        """Full pipeline: download → scrape retention → return asset."""
        video_id = self._extract_video_id(url)
        logger.info(f"[downloader] Starting download for {video_id}")

        # 1. Fetch rich metadata without downloading yet
        meta = self._fetch_metadata(url)

        # 2. Download video (best quality ≤1080p) + separate audio
        video_path = self._download_video(url, video_id)
        audio_path = self._extract_audio(video_path, video_id)

        # 3. Download subtitles (prefer en, fall back to auto-generated)
        subtitle_path = self._download_subtitles(url, video_id)

        # 4. Scrape Most-Replayed heatmap
        retention_graph = self._scrape_retention_graph(url)
        logger.info(
            f"[downloader] Retention graph: {len(retention_graph)} points"
        )

        asset = VideoAsset(
            video_id=video_id,
            title=meta.get("title", "Unknown"),
            description=meta.get("description", ""),
            duration_seconds=float(meta.get("duration", 0)),
            video_path=video_path,
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            thumbnail_url=meta.get("thumbnail", ""),
            channel_name=meta.get("uploader", "Unknown"),
            view_count=int(meta.get("view_count") or 0),
            retention_graph=retention_graph,
            raw_metadata=meta,
        )
        logger.success(f"[downloader] Asset ready: {asset.title!r}")
        return asset

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_video_id(url: str) -> str:
        parsed = urlparse(url)
        if parsed.hostname in ("youtu.be",):
            return parsed.path.lstrip("/")
        qs = parse_qs(parsed.query)
        vid = qs.get("v", [None])[0]
        if vid:
            return vid
        # short / embed paths
        match = re.search(r"(?:shorts|embed|v)/([A-Za-z0-9_-]{11})", url)
        if match:
            return match.group(1)
        raise ValueError(f"Cannot extract video ID from URL: {url}")

    def _fetch_metadata(self, url: str) -> dict:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writeinfojson": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info  # type: ignore[return-value]

    def _download_video(self, url: str, video_id: str) -> Path:
        out_path = self.output_dir / f"{video_id}_video.mp4"
        if out_path.exists():
            logger.info(f"[downloader] Video already cached: {out_path}")
            return out_path

        ydl_opts = {
            "format": (
                "bestvideo[height<=1080][ext=mp4]"
                "+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best"
            ),
            "outtmpl": str(out_path.with_suffix("")),  # yt-dlp adds ext
            "merge_output_format": "mp4",
            "quiet": False,
            "noprogress": False,
            "postprocessors": [{
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # yt-dlp may write <id>_video.mp4 directly
        candidate = self.output_dir / f"{video_id}_video.mp4"
        if not candidate.exists():
            # fallback: find any .mp4 with the video_id prefix
            found = list(self.output_dir.glob(f"{video_id}*.mp4"))
            if found:
                candidate = found[0]
        logger.debug(f"[downloader] Video path: {candidate}")
        return candidate

    def _extract_audio(self, video_path: Path, video_id: str) -> Path:
        audio_path = self.output_dir / f"{video_id}_audio.wav"
        if audio_path.exists():
            return audio_path
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le", str(audio_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.debug(f"[downloader] Audio extracted: {audio_path}")
        return audio_path

    def _download_subtitles(self, url: str, video_id: str) -> Optional[Path]:
        vtt_path = self.output_dir / f"{video_id}.en.vtt"
        if vtt_path.exists():
            return vtt_path

        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "vtt",
            "outtmpl": str(self.output_dir / video_id),
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Search for the VTT file
        candidates = list(self.output_dir.glob(f"{video_id}*.vtt"))
        if candidates:
            logger.debug(f"[downloader] Subtitles: {candidates[0]}")
            return candidates[0]
        logger.warning("[downloader] No subtitles found; will use Whisper/AssemblyAI")
        return None

    def _scrape_retention_graph(self, url: str) -> list[RetentionPoint]:
        """
        Scrape the ytHeatmapData from the page's ytInitialData blob.
        YouTube embeds heatmap data as JSON inside the page's <script> tags.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            with httpx.Client(timeout=20, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                html = resp.text

            # ytInitialData is a JS variable assignment
            match = re.search(
                r"var ytInitialData\s*=\s*(\{.+?\});\s*</script>",
                html, re.DOTALL
            )
            if not match:
                logger.warning("[downloader] ytInitialData not found in page")
                return []

            data = json.loads(match.group(1))
            return self._parse_heatmap(data)

        except Exception as exc:
            logger.warning(f"[downloader] Retention scrape failed: {exc}")
            return []

    @staticmethod
    def _parse_heatmap(data: dict) -> list[RetentionPoint]:
        """Walk the ytInitialData tree to find heatmapRenderer data."""
        points: list[RetentionPoint] = []
        try:
            # Path can shift; search recursively for heatmapRenderer
            raw = json.dumps(data)
            heatmap_match = re.findall(
                r'"heatMarkers"\s*:\s*(\[.+?\])\s*,\s*"heatMarkersDecorations"',
                raw
            )
            if not heatmap_match:
                return []

            markers = json.loads(heatmap_match[0])
            max_score = max(
                (m["heatMarkerRenderer"]["heatMarkerIntensityScoreNormalized"]
                 for m in markers),
                default=1.0,
            ) or 1.0

            for marker in markers:
                r = marker["heatMarkerRenderer"]
                intensity = (
                    r["heatMarkerIntensityScoreNormalized"] / max_score
                )
                points.append(RetentionPoint(
                    start_millis=int(r["timeRangeStartMillis"]),
                    duration_millis=int(r["markerDurationMillis"]),
                    intensity=round(intensity, 4),
                ))
        except Exception as exc:
            logger.warning(f"[downloader] Heatmap parse error: {exc}")

        return points


# ─── RSS Channel Monitor ──────────────────────────────────────────────────────

class ChannelMonitor:
    """Poll a YouTube channel's RSS feed for new uploads."""

    RSS_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

    def __init__(self, channel_id: str, seen_ids_path: Path):
        self.feed_url = self.RSS_TEMPLATE.format(channel_id=channel_id)
        self.seen_ids_path = seen_ids_path
        self._load_seen()

    def _load_seen(self):
        if self.seen_ids_path.exists():
            self._seen: set[str] = set(
                self.seen_ids_path.read_text().splitlines()
            )
        else:
            self._seen = set()

    def _save_seen(self):
        self.seen_ids_path.write_text("\n".join(sorted(self._seen)))

    def poll(self) -> list[str]:
        """Return list of new video URLs since last poll."""
        import xml.etree.ElementTree as ET
        ns = {"yt": "http://www.youtube.com/xml/schemas/2015"}

        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(self.feed_url)
                resp.raise_for_status()
        except Exception as exc:
            logger.error(f"[monitor] RSS fetch failed: {exc}")
            return []

        root = ET.fromstring(resp.text)
        new_urls: list[str] = []
        for entry in root.findall("entry", {"": "http://www.w3.org/2005/Atom"}):
            vid_id_el = entry.find("yt:videoId", ns)
            if vid_id_el is None:
                continue
            vid_id = vid_id_el.text or ""
            if vid_id not in self._seen:
                new_urls.append(f"https://www.youtube.com/watch?v={vid_id}")
                self._seen.add(vid_id)

        self._save_seen()
        logger.info(f"[monitor] Found {len(new_urls)} new videos")
        return new_urls
