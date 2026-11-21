"""
dashboard.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 (Optional) — Human-in-the-Loop Web Dashboard

Run with:  streamlit run dashboard.py

Features:
  • Auto-Pilot toggle — when ON, clips are published immediately by main.py
  • Manual Review queue — clips sit here for a 1-click approve / reject
  • Per-clip metadata preview (title, caption, virality score, hashtags)
  • Native video preview in-browser
  • Approve → triggers publisher.py; Reject → moves to trash
  • Live log tail from logs/pipeline.log
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="🎬 Viral Clip Pipeline",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Resolve project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from publisher import MultiPlatformPublisher
from analyzer import ViralClip
from downloader import VideoAsset

QUEUE_PATH = settings.output_dir / "queue.json"
TRASH_DIR = settings.output_dir / "trash"
TRASH_DIR.mkdir(parents=True, exist_ok=True)

publisher = MultiPlatformPublisher()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_queue() -> list[dict]:
    if not QUEUE_PATH.exists():
        return []
    try:
        return json.loads(QUEUE_PATH.read_text())
    except Exception:
        return []


def save_queue(queue: list[dict]) -> None:
    QUEUE_PATH.write_text(json.dumps(queue, indent=2))


def approve_clip(entry: dict) -> str:
    """Publish clip and return a status message."""
    clip_path = Path(entry["clip_path"])
    if not clip_path.exists():
        return f"❌ File not found: {clip_path}"

    try:
        clip = ViralClip(**entry["clip"])
        # We need minimal VideoAsset for back-link generation
        asset_stub = _stub_asset(entry)
        report = publisher.publish(clip_path, clip, asset_stub)

        successes = [r.platform for r in report.results if r.success]
        failures = [r.platform for r in report.results if not r.success]
        msg = ""
        if successes:
            msg += f"✅ Published to: {', '.join(successes)}. "
        if failures:
            msg += f"⚠️ Failed: {', '.join(failures)}."
        return msg or "⚠️ No platforms configured."
    except Exception as exc:
        return f"❌ Error: {exc}"


def _stub_asset(entry: dict) -> "VideoAsset":
    """Create a minimal VideoAsset from queue JSON for the publisher."""
    from downloader import VideoAsset
    from pathlib import Path
    return VideoAsset(
        video_id=entry.get("video_id", "unknown"),
        title=entry.get("video_title", ""),
        description="",
        duration_seconds=0,
        video_path=Path("/dev/null"),
        audio_path=Path("/dev/null"),
        subtitle_path=None,
        thumbnail_url="",
        channel_name="",
        view_count=0,
    )


def reject_clip(entry: dict) -> None:
    clip_path = Path(entry["clip_path"])
    if clip_path.exists():
        clip_path.rename(TRASH_DIR / clip_path.name)


def score_color(score: float) -> str:
    if score >= 8.5:
        return "🔴"
    elif score >= 7.0:
        return "🟠"
    else:
        return "🟡"


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/64px-YouTube_full-color_icon_%282017%29.svg.png", width=40)
    st.title("Viral Clip Pipeline")
    st.divider()

    # ── Auto-Pilot Toggle ─────────────────────────────────────────────────
    st.subheader("🤖 Pipeline Mode")
    auto_pilot = st.toggle(
        "Auto-Pilot",
        value=settings.auto_publish,
        help=(
            "When ON: clips are published automatically by main.py.\n"
            "When OFF: clips queue here for manual approval."
        ),
    )
    if auto_pilot != settings.auto_publish:
        settings.auto_publish = auto_pilot
        st.success(
            "Auto-Pilot " + ("enabled ✅" if auto_pilot else "disabled 🛑")
        )

    st.divider()

    # ── Quick trigger ─────────────────────────────────────────────────────
    st.subheader("▶️ Process a Video")
    url_input = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=…")
    col1, col2 = st.columns(2)
    if col1.button("🚀 Run Pipeline", use_container_width=True):
        if url_input:
            mode = "--auto" if auto_pilot else "--manual"
            st.info(f"Launching pipeline for:\n`{url_input}`")
            subprocess.Popen(
                [sys.executable, str(ROOT / "main.py"), "--url", url_input, mode],
                cwd=str(ROOT),
            )
            st.success("Pipeline started in background!")
        else:
            st.warning("Enter a URL first.")

    if col2.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    st.divider()

    # ── Settings summary ──────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    st.caption(f"**Model:** {settings.llm_model}")
    st.caption(f"**Max clips:** {settings.max_clips_per_video}")
    st.caption(f"**Min score:** {settings.min_clip_virality_score}/10")
    st.caption(f"**Resolution:** {settings.output_resolution[0]}×{settings.output_resolution[1]}")


# ─── Main Area ────────────────────────────────────────────────────────────────

st.title("🎬 Viral Clip Review Queue")

queue = load_queue()
pending = [e for e in queue if e.get("status") == "pending"]
published = [e for e in queue if e.get("status") == "published"]
rejected = [e for e in queue if e.get("status") == "rejected"]

col_p, col_pub, col_r = st.columns(3)
col_p.metric("⏳ Pending", len(pending))
col_pub.metric("✅ Published", len(published))
col_r.metric("🗑️ Rejected", len(rejected))

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_pending, tab_published, tab_logs = st.tabs(
    ["⏳ Pending Review", "✅ Published", "📋 Live Logs"]
)

# ── Pending Tab ───────────────────────────────────────────────────────────────
with tab_pending:
    if not pending:
        st.info(
            "No clips awaiting review.\n\n"
            "Run the pipeline via the sidebar or CLI:\n"
            "```\npython main.py --url <URL> --manual\n```"
        )
    else:
        for idx, entry in enumerate(pending):
            clip_data = entry.get("clip", {})
            score = clip_data.get("virality_score", 0)
            title = clip_data.get("vertical_title", "Untitled")
            hook = clip_data.get("hook_sentence", "")
            caption = clip_data.get("seo_caption", "")
            hashtags = clip_data.get("hashtags", [])
            clip_path = Path(entry.get("clip_path", ""))
            created = entry.get("created_at", "")

            with st.expander(
                f"{score_color(score)} [{score:.1f}/10]  {title}",
                expanded=(idx == 0),
            ):
                col_vid, col_meta = st.columns([1, 1])

                with col_vid:
                    if clip_path.exists():
                        st.video(str(clip_path))
                    else:
                        st.warning(f"File not found:\n`{clip_path}`")

                with col_meta:
                    st.markdown(f"**📺 Source video:** `{entry.get('video_id', '?')}`")
                    st.markdown(f"**📅 Created:** {created}")
                    st.markdown(f"**🎯 Hook:**\n> {hook}")
                    st.markdown("**📝 Caption:**")
                    st.text_area(
                        label="caption", value=caption,
                        height=120, label_visibility="collapsed",
                        key=f"caption_{idx}",
                    )
                    st.markdown(
                        "**#️⃣ Hashtags:** "
                        + " ".join(f"`#{t.lstrip('#')}`" for t in hashtags[:10])
                    )

                st.divider()
                col_approve, col_reject, _ = st.columns([1, 1, 3])

                if col_approve.button(
                    "✅ Approve & Publish", key=f"approve_{idx}",
                    type="primary", use_container_width=True
                ):
                    with st.spinner("Publishing…"):
                        msg = approve_clip(entry)
                    st.success(msg)
                    # Update queue
                    for q_entry in queue:
                        if q_entry is entry:
                            q_entry["status"] = "published"
                            q_entry["published_at"] = time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            )
                    save_queue(queue)
                    time.sleep(1)
                    st.rerun()

                if col_reject.button(
                    "🗑️ Reject", key=f"reject_{idx}",
                    use_container_width=True
                ):
                    reject_clip(entry)
                    for q_entry in queue:
                        if q_entry is entry:
                            q_entry["status"] = "rejected"
                    save_queue(queue)
                    st.rerun()

# ── Published Tab ─────────────────────────────────────────────────────────────
with tab_published:
    if not published:
        st.info("No published clips yet.")
    else:
        for entry in reversed(published):
            clip_data = entry.get("clip", {})
            score = clip_data.get("virality_score", 0)
            title = clip_data.get("vertical_title", "Untitled")
            st.markdown(
                f"**{score_color(score)} {title}** — "
                f"published {entry.get('published_at', '?')}"
            )

# ── Logs Tab ──────────────────────────────────────────────────────────────────
with tab_logs:
    log_path = settings.logs_dir / "pipeline.log"
    if log_path.exists():
        log_text = log_path.read_text(errors="replace")
        last_lines = "\n".join(log_text.splitlines()[-120:])
        st.code(last_lines, language="text")
        if st.button("🔄 Refresh logs"):
            st.rerun()
    else:
        st.info("No log file yet. Run the pipeline first.")
