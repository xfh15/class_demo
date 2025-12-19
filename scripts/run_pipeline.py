import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from ingest.audio_extractor import extract_audio
from asr.funasr_pipeline import FunASRPipeline, save_transcript
from analysis.gemini_client import GeminiClient
from report.report_builder import build_report, save_report


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classroom verbal analysis demo")
    parser.add_argument("--video", help="Input video file path")
    parser.add_argument("--video_url", help="Download video from URL before processing")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config YAML path")
    parser.add_argument("--workdir", default=None, help="Override workdir from config")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    workdir = Path(args.workdir or cfg.get("output", {}).get("workdir", "artifacts"))
    workdir.mkdir(parents=True, exist_ok=True)

    audio_path = workdir / "audio.wav"
    transcript_path = workdir / "transcript.json"
    report_path = workdir / "report.md"

    if not args.video and not args.video_url:
        parser.error("please provide --video or --video_url")
    video_path = None
    if args.video:
        video_path = Path(args.video)
    else:
        from ingest.downloader import download_video

        video_path = workdir / "download.mp4"
        print(f"[0/4] Downloading video from {args.video_url} to {video_path}")
        download_video(args.video_url, video_path)

    print(f"[1/4] Extracting audio to {audio_path}")
    extract_audio(video_path, audio_path, sample_rate=cfg["ffmpeg"]["sample_rate"], channels=cfg["ffmpeg"]["channels"])

    print("[2/4] Running FunASR (or mock)...")
    asr = FunASRPipeline(cfg.get("funasr", {}))
    utterances = asr.transcribe(audio_path)
    save_transcript(utterances, transcript_path)

    print("[3/4] Running Gemini analysis (or mock)...")
    gemini = GeminiClient(cfg.get("gemini", {}))
    analysis = gemini.analyze([u.__dict__ for u in utterances])

    print(f"[4/4] Building report at {report_path}")
    report_content = build_report([u.__dict__ for u in utterances], analysis)
    save_report(report_content, report_path)

    print("Done.")
    print(f"Audio: {audio_path}")
    print(f"Transcript: {transcript_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
