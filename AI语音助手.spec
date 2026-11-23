# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置 — AI 离线语音输入 & 对话
- onedir 免解压, windowed 无黑窗
- frontend.html 打包进内部
- version_info.txt 写入 Windows 文件版本属性
- models/ 不打包 (447MB), 运行时从 exe 同级目录读取
构建: pyinstaller AI语音助手.spec --noconfirm
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("frontend.html", "."), ("VoiceRecorderApp.png", ".")]
binaries = []
hiddenimports = []

# 收集需要携带二进制/数据的第三方库
for pkg in ["onnxruntime", "kaldi_native_fbank", "sentencepiece", "soxr", "webview",
            "kokoro", "misaki", "espeakng_loader", "torch", "spacy",
            "en_core_web_sm", "jieba", "phonemizer", "curated_tokenizers",
            "curated_transformers", "transformers", "thinc", "blis",
            "language_tags", "csvw", "segments"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# uvicorn / fastapi 动态导入
hiddenimports += collect_submodules("uvicorn")
hiddenimports += [
    "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
    "asr_engine", "llm_engine", "tts_engine", "db", "server", "paths", "version",
]


a = Analysis(
    ["gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL", "pandas", "scipy"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI语音助手",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,        # 无控制台黑窗
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="app.ico",
    version="version_info.txt",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="AI语音助手")
