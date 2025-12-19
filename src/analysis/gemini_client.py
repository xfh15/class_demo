import os
from typing import Dict, Any, List

try:
    import google.generativeai as genai
except Exception:
    genai = None


class GeminiClient:
    """
    Optional Gemini wrapper. Falls back to mock analysis if disabled or unavailable.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.enabled = cfg.get("enabled", False)
        self.model_name = cfg.get("model", "gemini-pro")
        self.prompt_path = cfg.get("prompt_path")
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._model = None

    def load(self) -> None:
        if not self.enabled:
            return
        if genai is None:
            raise RuntimeError("google-generativeai not installed. Set enabled=false to skip.")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)

    def analyze(self, utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.enabled:
            return self._mock_report()
        if self._model is None:
            self.load()
        prompt = self._build_prompt(utterances)
        response = self._model.generate_content(prompt)
        return {"report": response.text}

    def _build_prompt(self, utterances: List[Dict[str, Any]]) -> str:
        base = ""
        if self.prompt_path:
            base = open(self.prompt_path, "r", encoding="utf-8").read()
        transcript = "\n".join(
            f"[{u['start']:.2f}-{u['end']:.2f}] {u['speaker']}: {u['text']}"
            for u in utterances
        )
        return f"{base}\n\n转写文本：\n{transcript}\n\n请生成课堂状态报告（参与度、提问、主题偏移、建议）。"

    def _mock_report(self) -> Dict[str, Any]:
        return {
            "report": (
                "Mock 报告：学生参与度良好（S2 提出关键问题），主题聚焦能量守恒。"
                "建议保留互动提问，补充练习题。"
            )
        }
