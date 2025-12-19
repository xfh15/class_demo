import shutil
from pathlib import Path
from typing import Optional

import requests


def download_video(url: str, output_path: Path, timeout: Optional[int] = 30) -> None:
    """
    Download video/audio from URL to local path (streamed).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
