import json
from dataclasses import dataclass, asdict
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from funasr import AutoModel
except Exception:
    AutoModel = None


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str


class FunASRPipeline:
    """
    Thin wrapper around FunASR for ASR + diarization.
    Falls back to a mock result only when explicitly requested.
    """

    def __init__(self, cfg: Dict[str, Any]):
        lang = cfg.get("language")
        self.model_name = cfg.get("model") or self._select_model(lang)
        self.vad_model = cfg.get("vad_model")
        self.punc_model = cfg.get("punc_model")
        self.spk_model = cfg.get("spk_model")
        self.device = cfg.get("device", "cpu")
        self.user_wants_mock = cfg.get("use_mock", False)
        self.auto_model_available = AutoModel is not None
        if not self.user_wants_mock and not self.auto_model_available:
            raise RuntimeError("FunASR AutoModel unavailable. Install funasr and deps or set use_mock=true.")
        self.use_mock = self.user_wants_mock
        self.hotword = cfg.get("hotword")
        self._model = None

    def load(self) -> None:
        if self.use_mock:
            return
        if AutoModel is None:
            raise RuntimeError("FunASR is not installed. Set use_mock=true to bypass.")
        self._model = AutoModel(
            model=self.model_name,
            vad_model=self.vad_model,
            punc_model=self.punc_model,
            spk_model=self.spk_model,
            device=self.device,
        )

    def transcribe(self, audio_path: Path) -> List[Utterance]:
        if self.use_mock:
            return self._mock_result()
        if self._model is None:
            self.load()
        # FunASR generate API. Adjust keys according to your installed version.
        result = self._model.generate(
            input=str(audio_path),
            batch_size=1,
            hotword=self.hotword,
        )
        return self._parse_result(result)

    def _parse_result(self, result: Any) -> List[Utterance]:
        """
        Expected FunASR output format varies by model.
        Common structure:
        [
            {
                "timestamp": [[start, end], ...],
                "text": "....",
                "speaker": ["S0", "S1", ...]  # optional
            }
        ]
        Adjust this parser to your actual result keys.
        """
        if not result:
            return []
        entry = result[0]
        timestamps: List[List[float]] = entry.get("timestamp", [])
        text = entry.get("text", "")
        speakers: List[str] = entry.get("speaker", [])

        utterances: List[Utterance] = []
        if timestamps:
            segments = text.split()
            for idx, ts in enumerate(timestamps):
                speaker = speakers[idx] if idx < len(speakers) else "S0"
                utterances.append(
                    Utterance(
                        speaker=speaker,
                        start=float(ts[0]),
                        end=float(ts[1]),
                        text=segments[idx] if idx < len(segments) else text,
                    )
                )
        else:
            utterances.append(Utterance(speaker="S0", start=0.0, end=0.0, text=text))
        return utterances

    def _mock_result(self) -> List[Utterance]:
        # Small deterministic mock useful on macOS dev without FunASR.
        demo = [
            ("S1", 0.0, 3.2, "大家早上好，今天我们讨论能量守恒。"),
            ("S2", 3.3, 5.5, "老师我有个问题，动能怎么计算？"),
            ("S1", 5.6, 8.0, "很好，公式是一二mv平方。"),
        ]
        return [Utterance(*row) for row in demo]

    def _select_model(self, lang: Optional[str]) -> str:
        """
        Lightweight language-to-model mapping for Chinese/English only.
        Override with explicit model in config when needed.
        """
        if not lang:
            return "paraformer-zh"
        lang = lang.lower()
        model_map = {
            "zh": "paraformer-zh",
            "en": "paraformer-en",
        }
        if lang not in model_map:
            raise ValueError(f"Unsupported language '{lang}'. Only zh/en are supported.")
        return model_map[lang]


def save_transcript(utterances: List[Utterance], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(u) for u in utterances]
    path.write_text(json.dumps({"utterances": data}, ensure_ascii=False, indent=2), encoding="utf-8")
