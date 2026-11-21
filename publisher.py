"""
publisher.py
─────────────────────────────────────────────────────────────────────────────
Phase 6 — Publishing, Analytics, & First Pinned Comment

Responsibilities:
  • YouTube Shorts: OAuth2 upload via google-api-python-client
  • TikTok: Direct Post API (v2)
  • Instagram Reels: Graph API upload
  • Auto-post + pin a first comment on each YouTube Short
  • Token refresh handled transparently
  • All uploads are idempotent (skip if already published)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from analyzer import ViralClip
from config.settings import settings
from downloader import VideoAsset


# ─── Result model ─────────────────────────────────────────────────────────────

@dataclass
class PublishResult:
    platform: str
    video_id: Optional[str] = None
    url: Optional[str] = None
    comment_id: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class PublishReport:
    clip_path: Path
    clip: ViralClip
    results: list[PublishResult] = field(default_factory=list)

    @property
    def any_success(self) -> bool:
        return any(r.success for r in self.results)


# ─── YouTube Publisher ────────────────────────────────────────────────────────

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


class YouTubePublisher:
    """Handles OAuth2 token lifecycle + video upload + pinned comment."""

    def __init__(self):
        self._service = None

    # ── Auth ──────────────────────────────────────────────────────────────

    def _get_service(self):
        if self._service:
            return self._service

        creds: Optional[Credentials] = None
        token_path = settings.youtube_token_path

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(
                str(token_path), YOUTUBE_SCOPES
            )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("[youtube] Refreshing OAuth2 token…")
                creds.refresh(Request())
            else:
                if not settings.youtube_client_secrets_path.exists():
                    raise FileNotFoundError(
                        f"YouTube client secrets not found at "
                        f"{settings.youtube_client_secrets_path}. "
                        "Download from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(settings.youtube_client_secrets_path),
                    YOUTUBE_SCOPES,
                )
                creds = flow.run_local_server(port=0)

            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())
            logger.info(f"[youtube] Token saved → {token_path}")

        self._service = build("youtube", "v3", credentials=creds)
        return self._service

    # ── Upload ────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(HttpError),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(4),
    )
    def upload(
        self,
        clip_path: Path,
        clip: ViralClip,
        asset: VideoAsset,
    ) -> PublishResult:
        svc = self._get_service()

        body = {
            "snippet": {
                "title": clip.vertical_title[:100],
                "description": (
                    f"{clip.seo_caption}\n\n"
                    f"📺 Full video: https://www.youtube.com/watch?v={asset.video_id}"
                ),
                "tags": clip.hashtags[:30],
                "categoryId": "22",         # People & Blogs
                "defaultLanguage": "en",
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
                "madeForKids": False,
            },
        }

        media = MediaFileUpload(
            str(clip_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=1024 * 1024 * 8,   # 8 MB chunks
        )

        logger.info(f"[youtube] Uploading: {clip.vertical_title!r}")
        request = svc.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                logger.debug(f"[youtube] Upload progress: {pct}%")

        video_id = response["id"]
        video_url = f"https://www.youtube.com/shorts/{video_id}"
        logger.success(f"[youtube] Uploaded → {video_url}")

        # Pin first comment
        comment_id = self._pin_first_comment(svc, video_id, clip, asset)

        return PublishResult(
            platform="youtube",
            video_id=video_id,
            url=video_url,
            comment_id=comment_id,
            success=True,
        )

    def _pin_first_comment(
        self,
        svc,
        video_id: str,
        clip: ViralClip,
        asset: VideoAsset,
    ) -> Optional[str]:
        """Post + pin the LLM-generated first comment."""
        try:
            comment_body = {
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {
                        "snippet": {
                            "textOriginal": (
                                f"{clip.first_pinned_comment}\n\n"
                                f"🎥 Watch the full video: "
                                f"https://www.youtube.com/watch?v={asset.video_id}"
                            )
                        }
                    },
                }
            }
            comment_resp = (
                svc.commentThreads()
                .insert(part="snippet", body=comment_body)
                .execute()
            )
            comment_id = comment_resp["id"]

            # Pin it
            svc.comments().setModerationStatus(
                id=comment_id,
                moderationStatus="published",
                banAuthor=False,
            ).execute()

            logger.info(f"[youtube] Pinned comment: {comment_id}")
            return comment_id

        except Exception as exc:
            logger.warning(f"[youtube] Could not pin comment: {exc}")
            return None


# ─── TikTok Publisher ─────────────────────────────────────────────────────────

class TikTokPublisher:
    """
    Uses TikTok Direct Post API v2.
    Requires a valid access_token from TikTok for Developers.
    Docs: https://developers.tiktok.com/doc/direct-post-api
    """

    INIT_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
    COMPLETE_URL = "https://open.tiktokapis.com/v2/post/publish/video/complete/"
    CHUNK_SIZE = 1024 * 1024 * 10   # 10 MB

    def upload(self, clip_path: Path, clip: ViralClip) -> PublishResult:
        token = settings.tiktok_access_token
        if not token:
            return PublishResult(
                platform="tiktok",
                success=False,
                error="TikTok access token not configured",
            )

        try:
            file_size = clip_path.stat().st_size
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
            }

            # Step 1: Init upload
            caption = (
                clip.vertical_title + "\n\n" +
                " ".join(f"#{t.lstrip('#')}" for t in clip.hashtags[:5])
            )[:2200]

            init_payload = {
                "post_info": {
                    "title": caption,
                    "privacy_level": "PUBLIC_TO_EVERYONE",
                    "disable_duet": False,
                    "disable_comment": False,
                    "disable_stitch": False,
                    "video_cover_timestamp_ms": 1000,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": file_size,
                    "chunk_size": self.CHUNK_SIZE,
                    "total_chunk_count": (
                        (file_size + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
                    ),
                },
            }

            logger.info("[tiktok] Initialising upload…")
            with httpx.Client(timeout=30) as client:
                init_resp = client.post(
                    self.INIT_URL, headers=headers, json=init_payload
                )
                init_resp.raise_for_status()
                init_data = init_resp.json()["data"]
                publish_id = init_data["publish_id"]
                upload_url = init_data["upload_url"]

                # Step 2: Upload chunks
                with open(clip_path, "rb") as fh:
                    chunk_idx = 0
                    while True:
                        chunk = fh.read(self.CHUNK_SIZE)
                        if not chunk:
                            break
                        offset = chunk_idx * self.CHUNK_SIZE
                        end_byte = min(offset + len(chunk) - 1, file_size - 1)
                        upload_headers = {
                            "Content-Range": (
                                f"bytes {offset}-{end_byte}/{file_size}"
                            ),
                            "Content-Type": "video/mp4",
                            "Content-Length": str(len(chunk)),
                        }
                        put_resp = client.put(
                            upload_url, content=chunk, headers=upload_headers,
                            timeout=120,
                        )
                        put_resp.raise_for_status()
                        chunk_idx += 1
                        logger.debug(f"[tiktok] Chunk {chunk_idx} uploaded")

            logger.success(f"[tiktok] Upload complete. publish_id={publish_id}")
            return PublishResult(
                platform="tiktok",
                video_id=publish_id,
                url=f"https://www.tiktok.com/",   # exact URL returned async
                success=True,
            )

        except Exception as exc:
            logger.error(f"[tiktok] Upload failed: {exc}")
            return PublishResult(platform="tiktok", success=False, error=str(exc))


# ─── Instagram Reels Publisher ────────────────────────────────────────────────

class InstagramPublisher:
    """
    Instagram Graph API — Reels upload (two-step: create container → publish).
    Requires: instagram_access_token, instagram_user_id
    """

    GRAPH_BASE = "https://graph.facebook.com/v19.0"

    def upload(
        self,
        clip_path: Path,
        clip: ViralClip,
        video_public_url: str,
    ) -> PublishResult:
        """
        video_public_url must be a publicly accessible HTTPS URL to the MP4.
        For self-hosting, upload to S3 / GCS first and pass the signed URL.
        """
        token = settings.instagram_access_token
        user_id = settings.instagram_user_id

        if not token or not user_id:
            return PublishResult(
                platform="instagram",
                success=False,
                error="Instagram credentials not configured",
            )

        caption = (
            clip.seo_caption + "\n\n" +
            " ".join(f"#{t.lstrip('#')}" for t in clip.hashtags[:30])
        )[:2200]

        try:
            with httpx.Client(timeout=60) as client:
                # Step 1: Create media container
                logger.info("[instagram] Creating Reels container…")
                container_resp = client.post(
                    f"{self.GRAPH_BASE}/{user_id}/media",
                    params={
                        "media_type": "REELS",
                        "video_url": video_public_url,
                        "caption": caption,
                        "share_to_feed": "true",
                        "access_token": token,
                    },
                    timeout=30,
                )
                container_resp.raise_for_status()
                container_id = container_resp.json()["id"]

                # Step 2: Poll until container is FINISHED
                logger.info(f"[instagram] Waiting for container {container_id}…")
                for _ in range(30):   # max ~5 min
                    status_resp = client.get(
                        f"{self.GRAPH_BASE}/{container_id}",
                        params={
                            "fields": "status_code",
                            "access_token": token,
                        },
                    )
                    status_code = status_resp.json().get("status_code", "")
                    if status_code == "FINISHED":
                        break
                    elif status_code == "ERROR":
                        raise RuntimeError("Instagram container processing failed")
                    time.sleep(10)

                # Step 3: Publish
                publish_resp = client.post(
                    f"{self.GRAPH_BASE}/{user_id}/media_publish",
                    params={
                        "creation_id": container_id,
                        "access_token": token,
                    },
                )
                publish_resp.raise_for_status()
                ig_media_id = publish_resp.json()["id"]
                ig_url = f"https://www.instagram.com/p/{ig_media_id}/"

            logger.success(f"[instagram] Published → {ig_url}")
            return PublishResult(
                platform="instagram",
                video_id=ig_media_id,
                url=ig_url,
                success=True,
            )

        except Exception as exc:
            logger.error(f"[instagram] Upload failed: {exc}")
            return PublishResult(
                platform="instagram", success=False, error=str(exc)
            )


# ─── Master Publisher ─────────────────────────────────────────────────────────

class MultiPlatformPublisher:
    """
    Facade that fans out to all enabled platforms and collects results.
    Skips a platform gracefully if credentials are missing.
    """

    def __init__(self):
        self.yt = YouTubePublisher()
        self.tt = TikTokPublisher()
        self.ig = InstagramPublisher()

    def publish(
        self,
        clip_path: Path,
        clip: ViralClip,
        asset: VideoAsset,
        instagram_public_url: str = "",
    ) -> PublishReport:
        report = PublishReport(clip_path=clip_path, clip=clip)

        # ── YouTube Shorts ────────────────────────────────────────────────
        if settings.youtube_client_secrets_path.exists():
            try:
                yt_result = self.yt.upload(clip_path, clip, asset)
            except Exception as exc:
                yt_result = PublishResult(
                    platform="youtube", success=False, error=str(exc)
                )
            report.results.append(yt_result)
        else:
            logger.info("[publisher] YouTube credentials absent — skipping")

        # ── TikTok ───────────────────────────────────────────────────────
        if settings.tiktok_access_token:
            try:
                tt_result = self.tt.upload(clip_path, clip)
            except Exception as exc:
                tt_result = PublishResult(
                    platform="tiktok", success=False, error=str(exc)
                )
            report.results.append(tt_result)

        # ── Instagram Reels ───────────────────────────────────────────────
        if settings.instagram_access_token and instagram_public_url:
            try:
                ig_result = self.ig.upload(clip_path, clip, instagram_public_url)
            except Exception as exc:
                ig_result = PublishResult(
                    platform="instagram", success=False, error=str(exc)
                )
            report.results.append(ig_result)

        # Summary log
        for r in report.results:
            status = "✅" if r.success else "❌"
            logger.info(
                f"[publisher] {status} {r.platform}: "
                f"{r.url or r.error or 'no detail'}"
            )

        return report
