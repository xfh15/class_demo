# Classroom Verbal Analysis Demo

本地提取视频语音 → FunASR 转写（含说话人分离） → Gemini 语义分析 → 课堂状态报告。当前目录为空，已放好最小骨架，便于在 Linux/Windows 上直接扩展。

## 目录结构
- `scripts/run_pipeline.py`：CLI 流程入口（提取音频 → ASR → 分析 → 报告）。
- `src/ingest/audio_extractor.py`：ffmpeg 抽取音频。
- `src/asr/funasr_pipeline.py`：FunASR 本地推理封装（含占位 mock）。
- `src/analysis/gemini_client.py`：Gemini 调用/占位。
- `src/report/report_builder.py`：基于转写与分析结果生成简易报告。
- `configs/pipeline.yaml`：路径、模型、推理参数。
- `prompts/gemini_prompt.md`：Gemini 提示词示例。

## 先决条件（针对 Linux 服务器）
- Python 3.10+
- ffmpeg 可执行文件（放入 PATH）。
- 安装 FunASR 及模型：参考 https://github.com/alibaba-damo-academy/FunASR#installation
- Gemini 访问（可选）：在能联网的环境中配置 `GEMINI_API_KEY`（本地隐私文本即可仅在本机处理）。

## 快速开始（占位流程）
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# 运行示例（会在未安装 FunASR/Gemini 时自动走 mock）
python scripts/run_pipeline.py --video sample.mp4 --workdir artifacts
# 或指定视频 URL 下载后处理
python scripts/run_pipeline.py --video_url https://example.com/sample.mp4 --workdir artifacts
```

### 网页 UI（FastAPI）
```bash
uvicorn app.main:app --reload
# 打开 http://127.0.0.1:8000 上传视频/音频，查看报告
```
UI 默认走 mock，安装 FunASR / 配置 GEMINI_API_KEY 后自动使用真实推理。

### GPU 说明
- `configs/pipeline.yaml` 默认 `funasr.device: cuda:0`（服务器有 GPU 时）；若无 GPU 改为 `cpu`。

输出：
- `artifacts/audio.wav`：抽取的音频
- `artifacts/transcript.json`：转写+分离结果（若无 FunASR，则生成示例）
- `artifacts/report.md`：课堂状态报告（若无 Gemini，则生成示例）

## 关键配置
- `configs/pipeline.yaml`
  - `ffmpeg.sample_rate`：重采样为 ASR 期望采样率。
  - `funasr.language`：仅支持 `zh` 或 `en`，用于自动选择中/英模型；如需特定模型可直接填 `funasr.model`。
  - `funasr.model` / `vad_model` / `punc_model`：替换为本地模型名或路径。
  - `funasr.use_mock`：调试时关闭真实推理。
  - `gemini.enabled` / `model`：是否启用 Gemini。

## 调试建议
- 先在 Linux/WSL 跑通：`funasr` + `ffmpeg` 环境最稳定。
- 长音频可在 `funasr_pipeline.py` 中调整分片时长和重叠。
- 如果网络受限，可仅运行到转写阶段，后续报告由占位数据生成。

## 下一步可做
- 替换为真实 FunASR pipeline（去掉 mock），接入实际模型路径。
- 完善时间轴可视化/热力图（matplotlib/plotly）。
- 在 UI 层（FastAPI/Gradio）包装，支持上传视频文件。
