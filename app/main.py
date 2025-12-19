import uuid
from pathlib import Path
from typing import Dict, Any, List

import yaml
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from ingest.audio_extractor import extract_audio
from ingest.downloader import download_video
from asr.funasr_pipeline import FunASRPipeline
from analysis.gemini_client import GeminiClient
from report.report_builder import build_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs" / "pipeline.yaml"
load_dotenv(ROOT / ".env")
app = FastAPI(title="Classroom Verbal Analysis Demo")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(video_path: Path, cfg: Dict[str, Any], session_dir: Path) -> Dict[str, Any]:
    session_dir.mkdir(parents=True, exist_ok=True)
    audio_path = session_dir / "audio.wav"
    transcript_path = session_dir / "transcript.json"
    report_path = session_dir / "report.md"

    extract_audio(video_path, audio_path, sample_rate=cfg["ffmpeg"]["sample_rate"], channels=cfg["ffmpeg"]["channels"])

    asr = FunASRPipeline(cfg.get("funasr", {}))
    utterances = asr.transcribe(audio_path)

    gemini = GeminiClient(cfg.get("gemini", {}))
    analysis = gemini.analyze([u.__dict__ for u in utterances])

    report_content = build_report([u.__dict__ for u in utterances], analysis)
    report_path.write_text(report_content, encoding="utf-8")

    return {
        "utterances": [u.__dict__ for u in utterances],
        "analysis": analysis,
        "report_path": report_path,
        "report_content": report_content,
        "audio_path": audio_path,
        "transcript_path": transcript_path,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "error": None,
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, video: UploadFile = File(None), video_url: str = ""):
    cfg = load_config(DEFAULT_CFG)
    workdir = Path(cfg.get("output", {}).get("workdir", "artifacts"))
    session_dir = workdir / f"ui_{uuid.uuid4().hex[:8]}"
    session_dir.mkdir(parents=True, exist_ok=True)

    video_path = None
    if video and video.filename:
        video_ext = Path(video.filename).suffix or ".mp4"
        video_path = session_dir / f"upload{video_ext}"
        with open(video_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    elif video_url:
        video_ext = Path(video_url).suffix or ".mp4"
        video_path = session_dir / f"download{video_ext}"
        download_video(video_url, video_path)
    else:
        error = "请上传文件或提供视频地址"
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": error,
            },
        )

    result = None
    error = None
    try:
        result = run_pipeline(video_path, cfg, session_dir)
    except Exception as exc:
        error = str(exc)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "error": error,
        },
    )
