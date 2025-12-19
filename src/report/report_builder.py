from pathlib import Path
from typing import List, Dict, Any


def build_report(utterances: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
    lines = [
        "# 课堂语音分析报告",
        "",
        "## 概要",
        analysis.get("report", "未启用 Gemini，显示占位报告。"),
        "",
        "## 转写片段（含说话人分离）",
    ]
    for u in utterances:
        lines.append(
            f"- [{u['start']:.2f}-{u['end']:.2f}] {u['speaker']}: {u['text']}"
        )
    return "\n".join(lines)


def save_report(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
