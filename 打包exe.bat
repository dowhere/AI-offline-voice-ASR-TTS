@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo   打包 AI语音助手.exe
echo ============================================

if not exist ".venv\Scripts\python.exe" (
    echo 错误: 未找到虚拟环境, 请先运行 启动.bat
    pause & exit /b
)

echo 正在打包 (约 1-2 分钟)...
.venv\Scripts\pyinstaller.exe AI语音助手.spec --noconfirm

if exist "dist\AI语音助手.exe" (
    echo.
    echo ✓ 打包完成: dist\AI语音助手.exe
    echo.
    echo 分发说明: 把整个 models 文件夹复制到 exe 同级目录
    echo   dist\
    echo     AI语音助手.exe
    echo     models\Sensevoice-Small-ONNX\...
    echo.
) else (
    echo ✗ 打包失败, 请检查上方日志
)
pause
