"""
AI 离线语音输入 & 对话 — 桌面 GUI 启动器
启动 FastAPI 后端 + pywebview 原生窗口，一站式体验
"""
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.request

# uvicorn 日志静音
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from paths import APP_DIR
os.chdir(APP_DIR)

HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _probe_server(url, timeout=1.0):
    """判断 url 上是否已经运行着本应用"""
    try:
        with urllib.request.urlopen(f"{url}/api/status", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return isinstance(data, dict) and {"asr", "llm", "tts"} <= set(data)
    except Exception:
        return False


def _port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def start_server(port):
    """在后台线程启动 uvicorn 服务"""
    import uvicorn
    from server import app
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


def wait_for_server(url, timeout=30):
    """轮询等待服务就绪"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _probe_server(url, timeout=0.5):
            return True
        time.sleep(0.25)
    return False


def main():
    import webview

    default_url = f"http://{HOST}:{DEFAULT_PORT}"
    reused = _probe_server(default_url)

    if reused:
        # 已有实例在运行: 直接复用, 避免重复加载 1GB+ 模型
        print("检测到已运行的实例，复用现有服务...")
        url = default_url
    else:
        port = DEFAULT_PORT if _port_free(DEFAULT_PORT) else _find_free_port()
        url = f"http://{HOST}:{port}"
        # 1) 启动后端线程 (仅 init_db, 秒级就绪, 不加载模型)
        threading.Thread(target=start_server, args=(port,), daemon=True).start()
        # 2) 等待服务就绪 (很快, 因为不再阻塞加载引擎)
        print("正在启动 AI 语音助手...")
        if not wait_for_server(url):
            print("服务启动失败，请检查控制台输出")
            sys.exit(1)

    print(f"服务已就绪 ({url})，打开窗口...")

    # 3) 创建原生窗口
    window = webview.create_window(
        title="AI 离线语音输入 & 对话",
        url=url,
        width=1200,
        height=800,
        min_size=(900, 600),
        resizable=True,
        text_select=True,
    )

    # 4) 窗口显示后再触发模型加载 (界面未打开前不加载模型; 幂等)
    def _on_loaded():
        try:
            req = urllib.request.Request(f"{url}/api/engines/start", method="POST")
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    window.events.loaded += _on_loaded

    # 5) 运行事件循环（阻塞直到窗口关闭）
    #    Windows 使用 EdgeChromium(WebView2)，需允许麦克风等媒体权限
    webview.start(private_mode=False)

    # 6) 窗口关闭后退出
    print("窗口已关闭，退出程序")
    os._exit(0)


if __name__ == "__main__":
    main()
