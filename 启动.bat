@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo   AI 离线语音输入 ^& 对话  启动中...
echo ============================================
echo.

REM 检查虚拟环境
if not exist ".venv\Scripts\python.exe" (
    echo [1/2] 创建虚拟环境并安装依赖...
    where uv >nul 2>nul
    if %errorlevel%==0 (
        uv venv .venv --python 3.12
        uv pip install --python .venv\Scripts\python.exe -r requirements.txt
    ) else (
        python -m venv .venv
        .venv\Scripts\python.exe -m pip install -r requirements.txt
    )
)

echo [2/2] 启动桌面应用...
echo   (请确保 Ollama 已在运行: ollama serve)
echo.
.venv\Scripts\python.exe gui.py
