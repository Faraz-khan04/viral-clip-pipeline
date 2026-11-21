# 🎬 Autonomous YouTube → Shorts Viral Clip Pipeline

A fully autonomous, end-to-end system that ingests long-form YouTube videos,
identifies the most viral moments using multimodal AI analysis, edits them into
9:16 vertical shorts with kinetic captions, and publishes to YouTube Shorts,
TikTok, and Instagram Reels — with zero human intervention.

---

## Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        VIRAL CLIP PIPELINE                                   │
└──────────────────────────────────────────────────────────────────────────────┘

  YouTube URL / Channel RSS
          │
          ▼
  ┌───────────────────┐
  │  downloader.py    │  yt-dlp → MP4 + WAV + VTT
  │                   │  httpx  → Most-Replayed heatmap scrape
  └────────┬──────────┘
           │  VideoAsset(video_path, audio_path, retention_graph, metadata)
           ▼
  ┌───────────────────┐
  │  analyzer.py      │
  │                   │  AssemblyAI  → word-level transcript
  │                   │  librosa     → per-second RMS energy / spike detection
  │                   │  Claude      → Pydantic ViralClip structured outputs
  └────────┬──────────┘
           │  list[ViralClip]  (start/end/score/title/caption/overlays)
           ▼
  ┌───────────────────┐
  │  editor.py        │
  │                   │  ffmpeg      → extract raw clip
  │                   │  MediaPipe   → face detection per frame
  │                   │  OpenCV      → smooth 9:16 auto-crop / pan
  │                   │  Pexels API  → B-roll video inset
  │                   │  PIL         → kinetic word-by-word captions
  │                   │  ffmpeg      → loudness norm + H.264 final encode
  └────────┬──────────┘
           │  list[Path]  (finished 9:16 MP4s)
           ▼
  ┌───────────────────┐       ┌─────────────────────┐
  │  publisher.py     │  OR   │  queue.json          │
  │                   │       │  (manual review)     │
  │  YouTube Shorts   │       └──────────┬──────────┘
  │  TikTok           │                  │
  │  Instagram Reels  │         dashboard.py (Streamlit)
  │  + pinned comment │         1-click approve/reject
  └───────────────────┘
```

---

## Quick Start

### 1. System dependencies

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg redis-server
sudo fc-cache -fv   # rebuild font cache after installing fonts

# macOS
brew install ffmpeg redis
brew services start redis
```

### 2. Python environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env and add your API keys
```

#### Required keys
| Key | Where to get it |
|-----|----------------|
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `ASSEMBLYAI_API_KEY` | assemblyai.com |

#### Optional keys (enables publishing)
| Key | Where to get it |
|-----|----------------|
| `YOUTUBE_CLIENT_SECRETS_PATH` | Google Cloud Console → OAuth 2.0 client |
| `TIKTOK_ACCESS_TOKEN` | TikTok for Developers → Direct Post API |
| `INSTAGRAM_ACCESS_TOKEN` + `INSTAGRAM_USER_ID` | Meta for Developers → Instagram Graph API |
| `PEXELS_API_KEY` | pexels.com/api |

### 4. YouTube OAuth (first run only)

```bash
# Place your downloaded client_secrets.json here:
mkdir -p config
mv ~/Downloads/client_secret_*.json config/yt_client_secrets.json
```

The first run will open a browser for OAuth consent. Token is cached in
`config/yt_token.json` and auto-refreshed thereafter.

---

## Usage

### Process a single video (auto-publish)
```bash
python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --auto
```

### Process + queue for manual review
```bash
python main.py --url "https://youtu.be/..." --manual
streamlit run dashboard.py   # open http://localhost:8501
```

### Monitor a channel (runs forever, polls every 15 min)
```bash
python main.py --channel UCxxxxxxxxxxxxxxxxxxxxxx --interval 900
```

### Celery distributed mode
```bash
# Start Redis first, then:
celery -A celery_tasks.tasks worker --loglevel=info --concurrency=2

# Enqueue a video from Python:
from celery_tasks.tasks import process_video
process_video.delay("https://www.youtube.com/watch?v=...")
```

---

## Project Structure

```
viral_pipeline/
├── main.py                 ← Master orchestrator + CLI
├── downloader.py           ← Phase 1: yt-dlp + retention scrape
├── analyzer.py             ← Phase 2: Transcription + LLM virality scoring
├── editor.py               ← Phase 3/4: Face tracking + captions + encoding
├── publisher.py            ← Phase 6: YouTube/TikTok/Instagram + pinned comment
├── dashboard.py            ← Optional: Streamlit review UI
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.py         ← Pydantic-Settings config singleton
│   ├── yt_client_secrets.json   (you provide)
│   └── yt_token.json            (auto-generated on first run)
├── celery_tasks/
│   └── tasks.py            ← Celery async task definitions
├── output/
│   ├── source/             ← Downloaded source videos
│   ├── clips/              ← Finished 9:16 MP4s
│   ├── assets/broll/       ← Cached Pexels B-roll
│   └── queue.json          ← Manual review queue
└── logs/
    └── pipeline.log
```

---

## Key Design Decisions

| Concern | Solution |
|---------|----------|
| Structured LLM output | `pydantic` v2 models + Claude JSON mode |
| Face-tracking smoothness | Exponential moving average on crop-centre X |
| Rate limit resilience | `tenacity` retry with exponential backoff on all API calls |
| Token auto-refresh | `google-auth` `Request()` refresh + `yt_token.json` |
| Idempotency | File existence checks before re-downloading or re-rendering |
| Modularity | Each phase is a self-contained class; swap any module independently |
| Observability | `loguru` with rotating file log + `rich` progress bars |
| Scalability | Optional Celery + Redis for multi-worker distributed processing |

---

## Extending the Pipeline

- **Swap LLM provider**: change `llm_provider=openai` in `.env` and update the
  `LLMViralityAnalyser._client` instantiation in `analyzer.py`.
- **Add a new platform**: implement a class with an `upload()` method returning
  `PublishResult`, then register it in `MultiPlatformPublisher.publish()`.
- **Custom caption style**: all styling constants live in `config/settings.py`
  under the `Caption styling` section.
- **AI image B-roll**: replace `BRollFetcher` with a call to DALL-E or
  Stable Diffusion; the interface is identical (returns `Path` to a video/image).
