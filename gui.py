"""
AI 离线语音输入 & 对话 — 桌面 GUI 启动器
启动 FastAPI 后端 + pywebview 原生窗口，一站式体验
"""
import os
import sys
import threading
import time
import signal
import logging
import atexit

# uvicorn 日志静音
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

_DIR = os.path.dirname(os.path.abspath(__file__))
from paths import APP_DIR
os.chdir(APP_DIR)


def start_server():
    """在后台线程启动 uvicorn 服务"""
    import uvicorn
    from server import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


def wait_for_server(url="http://127.0.0.1:8000", timeout=30):
    """轮询等待服务就绪"""
    import urllib.request
    for i in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    import webview

    # 1) 启动后端线程
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    # 2) 等待服务就绪
    print("正在启动 AI 语音助手...")
    if not wait_for_server():
        print("服务启动失败，请检查控制台输出")
        sys.exit(1)

    print("服务已就绪，打开窗口...")

    # 3) 创建原生窗口
    window = webview.create_window(
        title="AI 离线语音输入 & 对话",
        url="http://127.0.0.1:8000",
        width=1200,
        height=800,
        min_size=(900, 600),
        resizable=True,
        text_select=True,
    )

    # 4) 运行事件循环（阻塞直到窗口关闭）
    #    Windows 使用 EdgeChromium(WebView2)，需允许麦克风等媒体权限
    webview.start(private_mode=False)

    # 5) 窗口关闭后退出
    print("窗口已关闭，退出程序")
    os._exit(0)


if __name__ == "__main__":
    main()