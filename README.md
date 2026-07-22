# AI 离线语音输入 & 对话

完全离线的语音识别 + 大语言模型对话应用。

## 功能

- **🎙️ 语音识别** — 浏览器实时录音识别 / 上传音频文件识别
- **📝 识别记录** — 自动保存识别历史，支持导出文本
- **💬 AI 对话** — 本地大模型对话，支持语音输入到对话
- **🔒 完全离线** — ASR 使用 SenseVoice-Small ONNX，LLM 使用本地 Ollama

## 支持的语种

中文、英文、粤语、日语、韩语，自动语种检测 + 情感识别

## 快速开始

### 1. 环境要求

- Python 3.10+
- [Ollama](https://ollama.com/) (已安装并运行)
- 至少一个本地大模型 (推荐 `qwen3:8b` 或 `qwen3.5:9b`)

### 2. 安装依赖

```bash
# 创建虚拟环境
uv venv .venv --python 3.12
# 或: python -m venv .venv

# 安装依赖
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. 启动 Ollama

```bash
ollama serve
```

确保已有模型：
```bash
ollama list
```

### 4. 启动应用

**桌面 GUI（推荐）** — 双击即用，启动后直接弹出原生窗口：

```bash
# 方式一: 双击 启动.bat            (带命令行窗口，可看日志)
# 方式二: 双击 启动-无黑窗.bat     (仅显示应用窗口，无黑框)
# 方式三: 命令行
.venv\Scripts\python.exe gui.py
```

**浏览器模式** — 仅启动后端，用浏览器访问：

```bash
.venv\Scripts\python.exe server.py
# 然后浏览器访问 http://127.0.0.1:8000
```

## 打包成 EXE（免安装分发）

双击 `打包exe.bat`（或运行 `pyinstaller AI语音助手.spec --noconfirm`），
在 `dist\` 生成单文件 `AI语音助手.exe`（约 47MB）。

**⚠️ 重要：分发时必须把 `models` 文件夹放到 exe 同级目录**（模型 447MB 不打包进 exe）：

```
分发目录/
├── AI语音助手.exe
└── models/
    └── Sensevoice-Small-ONNX/
        ├── SenseVoice-Encoder.fp16.onnx
        ├── SenseVoice-CTC.fp16.onnx
        └── tokenizer.bpe.model
```

双击 `AI语音助手.exe` 即启动（首次启动需解压依赖，约 5-10 秒）。
运行时会在 exe 同级目录生成 `data.db` 保存记录与对话。

> 注：对话功能仍需本机 Ollama 运行；语音识别完全独立、无需 Ollama。

## 项目结构

```
.
├── gui.py              # 桌面 GUI 启动器 (pywebview 原生窗口)
├── server.py           # FastAPI 后端
├── asr_engine.py       # SenseVoice ONNX 语音识别引擎
├── llm_engine.py       # Ollama LLM 对话引擎
├── db.py               # SQLite 存储层
├── frontend.html       # 单页 Web 前端
├── requirements.txt    # Python 依赖
├── 启动.bat            # 启动脚本 (带日志窗口)
├── 启动-无黑窗.bat     # 静默启动脚本
└── models/
    └── Sensevoice-Small-ONNX/
        ├── SenseVoice-Encoder.fp16.onnx
        ├── SenseVoice-CTC.fp16.onnx
        └── tokenizer.bpe.model
```

## 配置

### 切换大模型

在 `llm_engine.py` 中修改 `DEFAULT_MODEL`，或修改 `server.py` 中 `llm_engine.py` 的调用参数。

### 语种选择

前端顶栏可选择语种：自动/中文/英文/粤语/日语/韩语

## 技术栈

| 组件 | 技术 |
|------|------|
| 语音识别 | SenseVoice-Small ONNX |
| 特征提取 | kaldi-native-fbank + LFR |
| 解码 | CTC greedy + SentencePiece BPE |
| 大模型 | Ollama (qwen3:8b / 任意本地模型) |
| 后端 | FastAPI + Uvicorn |
| 存储 | SQLite |
| 前端 | 单页 HTML + Web Audio API |