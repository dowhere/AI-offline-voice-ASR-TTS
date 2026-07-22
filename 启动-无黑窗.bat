@echo off
REM 静默启动 (无命令行黑窗)，双击后仅显示 GUI 窗口
cd /d %~dp0
if not exist ".venv\Scripts\pythonw.exe" (
    echo 首次使用请先运行 启动.bat 安装依赖
    pause
    exit /b
)
start "" ".venv\Scripts\pythonw.exe" gui.py
