"""
统一路径管理 — 兼容源码运行与 PyInstaller 打包
- APP_DIR: exe/脚本所在目录，存放外部可写资源 (models/, data.db)
- resource_path(): 打包进 exe 内部的只读资源 (frontend.html)
"""
import os
import sys

if getattr(sys, "frozen", False):
    # PyInstaller 打包后: exe 实际所在目录
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))


def resource_path(rel):
    """返回打包资源路径 (onefile 时在 _MEIPASS 临时目录, 源码时在 APP_DIR)"""
    base = getattr(sys, "_MEIPASS", APP_DIR)
    return os.path.join(base, rel)
