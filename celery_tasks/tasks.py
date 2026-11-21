"""
celery_tasks/tasks.py
─────────────────────────────────────────────────────────────────────────────
Celery task definitions — allows the pipeline to be driven by a distributed
task queue (Redis broker) instead of a blocking subprocess.

Start worker:
  celery -A celery_tasks.tasks worker --loglevel=info --concurrency=2

Enqueue from Python:
  from celery_tasks.tasks import process_video
  process_video.delay("https://www.youtube.com/watch?v=...")

Beat scheduler (periodic channel poll):
  celery -A celery_tasks.tasks beat --loglevel=info
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when worker boots
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from celery import Celery
from celery.schedules import crontab
from loguru import logger

from config.settings import settings

# ── App init ──────────────────────────────────────────────────────────────────
app = Celery(
    "viral_pipeline",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,          # one task at a time per worker
    task_soft_time_limit=3600,             # 1 hr soft limit
    task_time_limit=4200,                  # 70 min hard limit
)


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    name="viral_pipeline.process_video",
)
def process_video(self, url: str, manual: bool = False) -> dict:
    """
    Full pipeline for a single YouTube URL.
    Returns a summary dict of published URLs.
    """
    try:
        from main import ViralPipeline
        pipeline = ViralPipeline(manual_review=manual)
        reports = pipeline.run(url)

        summary = []
        for rpt in reports:
            for r in rpt.results:
                summary.append({
                    "platform": r.platform,
                    "success": r.success,
                    "url": r.url,
                    "error": r.error,
                })
        return {"url": url, "results": summary}

    except Exception as exc:
        logger.error(f"[celery] Task failed for {url}: {exc}")
        raise self.retry(exc=exc)


@app.task(
    name="viral_pipeline.poll_channel",
    ignore_result=True,
)
def poll_channel(channel_id: str) -> None:
    """Polls a channel RSS and enqueues process_video for each new upload."""
    from downloader import ChannelMonitor
    monitor = ChannelMonitor(
        channel_id=channel_id,
        seen_ids_path=settings.output_dir / "seen_ids.txt",
    )
    new_urls = monitor.poll()
    for url in new_urls:
        logger.info(f"[celery] Enqueuing: {url}")
        process_video.delay(url)


# ── Beat schedule (optional periodic tasks) ───────────────────────────────────
# Edit WATCHED_CHANNELS in your .env or hard-code channel IDs here.

WATCHED_CHANNELS: list[str] = []  # e.g. ["UCxxxxxx", "UCyyyyyy"]

if WATCHED_CHANNELS:
    app.conf.beat_schedule = {
        f"poll-{ch}": {
            "task": "viral_pipeline.poll_channel",
            "schedule": crontab(minute="*/15"),   # every 15 min
            "args": (ch,),
        }
        for ch in WATCHED_CHANNELS
    }
