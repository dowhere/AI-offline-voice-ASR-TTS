@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo   打包 AI语音助手 (onedir 免解压)
echo ============================================

if not exist ".venv\Scripts\python.exe" (
    echo 错误: 未找到虚拟环境, 请先运行 启动.bat
    pause & exit /b
)

echo 正在打包 (约 5-10 分钟)...
.venv\Scripts\pyinstaller.exe AI语音助手.spec --noconfirm

if exist "dist\AI语音助手\AI语音助手.exe" (
    echo.
    echo ✓ 打包完成: dist\AI语音助手\AI语音助手.exe
    echo.
    echo 分发说明: 把整个文件夹发给用户
    echo   dist\AI语音助手\   (整个目录)
    echo     └─ models\       (手动放到 exe 旁边)
    echo.
    echo 双击 dist\AI语音助手\AI语音助手.exe 即可直接启动
    echo.
) else (
    echo ✗ 打包失败, 请检查上方日志
    findstr "Error" build\AI语音助手\warn-AI语音助手.txt 2>nul
)
pause