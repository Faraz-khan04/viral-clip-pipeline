"""
main.py
─────────────────────────────────────────────────────────────────────────────
Master Orchestrator — ties all phases together into a single autonomous loop.

Modes:
  • python main.py --url "https://youtube.com/watch?v=..."
    → Process a single video immediately

  • python main.py --channel CHANNEL_ID --interval 900
    → Poll a channel RSS every N seconds for new uploads

  • python main.py --url "..." --manual
    → Skip auto-publish; clips sit in output/clips/ for human review

All stages are wrapped with aggressive exception handling.
Token-rate-limit backoff is handled per-module via tenacity decorators.
Progress is logged to logs/pipeline.log and to stdout via rich.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn,
)
from rich.table import Table

from analyzer import VideoAnalyser, ViralClip
from config.settings import settings
from downloader import ChannelMonitor, VideoAsset, VideoDownloader
from editor import VideoEditor
from publisher import MultiPlatformPublisher, PublishReport


# ─── Logging Setup ────────────────────────────────────────────────────────────

settings.ensure_dirs()
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{message}</cyan>"
    ),
    colorize=True,
)
logger.add(
    settings.logs_dir / "pipeline.log",
    rotation="50 MB",
    retention="14 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)

console = Console()


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class ViralPipeline:
    """
    End-to-end pipeline for a single YouTube URL.
    Each stage is isolated so a failure in one clip never aborts others.
    """

    def __init__(self, manual_review: bool = False):
        self.manual_review = manual_review
        self.downloader = VideoDownloader(settings.output_dir / "source")
        self.analyser = VideoAnalyser()
        self.publisher = MultiPlatformPublisher()

    # ── Entry point ───────────────────────────────────────────────────────

    def run(self, url: str) -> list[PublishReport]:
        reports: list[PublishReport] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:

            # ── Phase 1: Download ─────────────────────────────────────────
            task_dl = progress.add_task("📥 Downloading video…", total=None)
            try:
                asset = self.downloader.download(url)
                progress.update(task_dl, description="✅ Download complete", total=1, completed=1)
            except Exception:
                progress.update(task_dl, description="❌ Download failed")
                logger.error(f"[main] Download failed:\n{traceback.format_exc()}")
                return []

            # ── Phase 2: Analyse ──────────────────────────────────────────
            task_an = progress.add_task("🧠 Analysing virality…", total=None)
            try:
                clips, words = self.analyser.run(asset)
                progress.update(
                    task_an,
                    description=f"✅ Found {len(clips)} clips",
                    total=1, completed=1,
                )
            except Exception:
                progress.update(task_an, description="❌ Analysis failed")
                logger.error(f"[main] Analysis failed:\n{traceback.format_exc()}")
                return []

            if not clips:
                logger.warning("[main] No clips met the virality threshold. Done.")
                return []

            self._print_clip_table(clips)

            # ── Phase 3 & 4: Edit each clip ───────────────────────────────
            editor = VideoEditor(asset, words)
            clip_paths: list[tuple[ViralClip, Path]] = []

            task_ed = progress.add_task(
                "✂️  Editing clips…", total=len(clips)
            )
            for idx, clip in enumerate(clips, start=1):
                try:
                    clip_path = editor.render_clip(clip, idx)
                    clip_paths.append((clip, clip_path))
                    progress.advance(task_ed)
                except Exception:
                    logger.error(
                        f"[main] Render failed for clip {idx}:\n"
                        f"{traceback.format_exc()}"
                    )
                    progress.advance(task_ed)
                    continue

            # ── Phase 5: Publish / queue ──────────────────────────────────
            if self.manual_review:
                logger.info(
                    "[main] Manual-review mode — clips are in "
                    f"{settings.output_dir / 'clips'}. "
                    "Run the Streamlit dashboard to approve & publish."
                )
                # Write a queue JSON for the dashboard to pick up
                self._write_queue(asset, clip_paths)
                return []

            if not settings.auto_publish:
                logger.info(
                    "[main] auto_publish=False — set it in .env or pass --auto "
                    "to enable automated publishing."
                )
                self._write_queue(asset, clip_paths)
                return []

            task_pub = progress.add_task(
                "🚀 Publishing…", total=len(clip_paths)
            )
            for clip, clip_path in clip_paths:
                try:
                    report = self.publisher.publish(clip_path, clip, asset)
                    reports.append(report)
                except Exception:
                    logger.error(
                        f"[main] Publish failed for {clip_path.name}:\n"
                        f"{traceback.format_exc()}"
                    )
                finally:
                    progress.advance(task_pub)

        self._print_publish_summary(reports)
        return reports

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _print_clip_table(clips: list[ViralClip]) -> None:
        table = Table(title="🎬 Identified Viral Clips", show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Score", justify="center", style="bold green", width=7)
        table.add_column("Window", style="cyan", width=16)
        table.add_column("Title", style="white")
        table.add_column("Hook", style="yellow")

        for i, c in enumerate(clips, 1):
            table.add_row(
                str(i),
                f"{c.virality_score:.1f}",
                f"{c.start_seconds:.0f}s → {c.end_seconds:.0f}s",
                c.vertical_title[:50],
                c.hook_sentence[:60] + "…",
            )
        console.print(table)

    @staticmethod
    def _print_publish_summary(reports: list[PublishReport]) -> None:
        if not reports:
            return
        table = Table(title="📊 Publish Summary", show_lines=True)
        table.add_column("Clip", style="dim")
        table.add_column("Platform", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("URL / Error", style="white")

        for rpt in reports:
            for r in rpt.results:
                table.add_row(
                    rpt.clip_path.name[:30],
                    r.platform,
                    "✅" if r.success else "❌",
                    r.url or r.error or "—",
                )
        console.print(table)

    @staticmethod
    def _write_queue(
        asset: VideoAsset,
        clip_paths: list[tuple[ViralClip, Path]],
    ) -> None:
        """Write a queue.json that the Streamlit dashboard reads."""
        queue_path = settings.output_dir / "queue.json"
        existing: list[dict] = []
        if queue_path.exists():
            try:
                existing = json.loads(queue_path.read_text())
            except Exception:
                pass

        import json
        for clip, path in clip_paths:
            existing.append({
                "video_id": asset.video_id,
                "video_title": asset.title,
                "clip_path": str(path),
                "clip": clip.model_dump(),
                "status": "pending",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

        queue_path.write_text(json.dumps(existing, indent=2))
        logger.info(f"[main] Queue written → {queue_path}")


# ─── Channel Monitor Loop ─────────────────────────────────────────────────────

class ChannelLoop:
    """Polls a YouTube channel RSS and runs the pipeline on every new upload."""

    def __init__(
        self,
        channel_id: str,
        interval_seconds: int = 900,
        manual_review: bool = False,
    ):
        self.monitor = ChannelMonitor(
            channel_id=channel_id,
            seen_ids_path=settings.output_dir / "seen_ids.txt",
        )
        self.pipeline = ViralPipeline(manual_review=manual_review)
        self.interval = interval_seconds

    def run_forever(self) -> None:
        logger.info(
            f"[channel_loop] Monitoring channel. "
            f"Poll interval: {self.interval}s"
        )
        while True:
            try:
                new_urls = self.monitor.poll()
                for url in new_urls:
                    logger.info(f"[channel_loop] New video detected: {url}")
                    try:
                        self.pipeline.run(url)
                    except Exception:
                        logger.error(
                            f"[channel_loop] Pipeline error for {url}:\n"
                            f"{traceback.format_exc()}"
                        )
            except Exception:
                logger.error(
                    f"[channel_loop] Poll error:\n{traceback.format_exc()}"
                )

            logger.info(
                f"[channel_loop] Sleeping {self.interval}s until next poll…"
            )
            time.sleep(self.interval)


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

import json


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autonomous YouTube → Shorts viral clip pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
# Process a single video:
  python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Process + skip auto-publish (manual review via dashboard):
  python main.py --url "https://youtu.be/..." --manual

# Monitor a channel every 15 minutes:
  python main.py --channel UCxxxxxxxxxxxxxx --interval 900

# Force auto-publish regardless of .env setting:
  python main.py --url "..." --auto
""",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", metavar="URL", help="Single YouTube video URL")
    g.add_argument(
        "--channel", metavar="CHANNEL_ID",
        help="YouTube channel ID to monitor via RSS"
    )
    p.add_argument(
        "--interval", type=int, default=900, metavar="SEC",
        help="RSS poll interval in seconds (default: 900)"
    )
    p.add_argument(
        "--manual", action="store_true",
        help="Queue clips for manual review instead of publishing"
    )
    p.add_argument(
        "--auto", action="store_true",
        help="Force auto-publish even if AUTO_PUBLISH=false in .env"
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.auto:
        settings.auto_publish = True

    console.rule("[bold red]🎬 Viral Clip Pipeline[/bold red]")
    console.print(
        f"[dim]Output dir:[/dim] {settings.output_dir}\n"
        f"[dim]Auto-publish:[/dim] {settings.auto_publish}\n"
        f"[dim]Manual review:[/dim] {args.manual}\n"
        f"[dim]LLM model:[/dim] {settings.llm_model}\n"
        f"[dim]Max clips:[/dim] {settings.max_clips_per_video}\n"
        f"[dim]Min virality score:[/dim] {settings.min_clip_virality_score}/10\n"
    )

    if args.url:
        pipeline = ViralPipeline(manual_review=args.manual)
        pipeline.run(args.url)
    else:
        loop = ChannelLoop(
            channel_id=args.channel,
            interval_seconds=args.interval,
            manual_review=args.manual,
        )
        loop.run_forever()


if __name__ == "__main__":
    main()
