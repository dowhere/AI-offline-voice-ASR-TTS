"""
应用版本信息 — 单一来源
修改版本时只需更新本文件与 version_info.txt（Windows 文件属性）
"""

APP_NAME = "AI 离线语音输入 & 对话"
APP_SHORT_NAME = "AI 离线语音助手"
APP_ID = "AI语音助手"
COMPANY = "zq"

__version__ = "1.1.0"
VERSION_TUPLE = (1, 1, 0, 0)

# 便于命令行/脚本读取
if __name__ == "__main__":
    print(__version__)
