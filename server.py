"""
FastAPI 后端服务
- /api/asr/upload: 上传音频文件识别
- /api/asr/record: 接收浏览器录音(PCM16)
- /api/transcripts: 识别记录 CRUD
- /api/chat: LLM 对话(流式 SSE)
- /api/conversations: 会话管理
- Web 前端静态文件
"""
import os
import json
import io
import uuid
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from asr_engine import get_engine, ASREngine
from llm_engine import get_engine as get_llm, LLMEngine
from db import init_db, add_transcript, list_transcripts, delete_transcript, clear_transcripts
from db import create_conversation, list_conversations, rename_conversation, delete_conversation
from db import add_message, get_messages
from paths import APP_DIR, resource_path

_DIR = APP_DIR

app = FastAPI(title="AI 离线语音输入 & 对话", version="1.0.0")

# ---------- 启动时 ----------
@app.on_event("startup")
def startup():
    init_db()
    print("[启动] 数据库初始化完成")
    # 预加载 ASR 引擎
    get_engine()
    # 预加载 LLM 引擎
    get_llm()


# ---------- ASR 接口 ----------
@app.post("/api/asr/upload")
async def asr_upload(file: UploadFile = File(...), language: str = Form("auto")):
    content = await file.read()
    if not content:
        raise HTTPException(400, "空文件")
    # 保存临时文件, 用 ffmpeg 统一转为 16kHz 16bit WAV
    ext = os.path.splitext(file.filename)[1] or ".wav"
    tmp_in = os.path.join(tempfile.gettempdir(), f"asr_{uuid.uuid4().hex}{ext}")
    tmp_out = os.path.join(tempfile.gettempdir(), f"asr_{uuid.uuid4().hex}.wav")
    try:
        with open(tmp_in, "wb") as f:
            f.write(content)
        # 转码
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp_out],
            capture_output=True, timeout=30,
        )
        result = get_engine().transcribe_file(tmp_out, language=language)
        result["source"] = "upload"
        # 保存到数据库
        result["id"] = add_transcript(
            text=result["text"], language=result["language"],
            emotion=result["emotion"], duration=result["duration"],
            source="upload",
        )
        return result
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "音频转码超时")
    except Exception as e:
        raise HTTPException(500, f"识别失败: {e}")
    finally:
        for p in (tmp_in, tmp_out):
            if os.path.exists(p):
                os.remove(p)


@app.post("/api/asr/record")
async def asr_record(request: Request, sample_rate: int = 16000, language: str = "auto"):
    """接收浏览器录音的 PCM16 原始字节流 (application/octet-stream)"""
    data = await request.body()
    if not data:
        raise HTTPException(400, "空数据")
    try:
        result = get_engine().transcribe_pcm16(data, sample_rate, language=language)
        result["source"] = "record"
        result["id"] = add_transcript(
            text=result["text"], language=result["language"],
            emotion=result["emotion"], duration=result["duration"],
            source="record",
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"识别失败: {e}")


@app.get("/api/transcripts")
def api_list_transcripts(limit: int = 200):
    return list_transcripts(limit)


@app.delete("/api/transcripts/{tid}")
def api_delete_transcript(tid: int):
    delete_transcript(tid)
    return {"ok": True}


@app.delete("/api/transcripts")
def api_clear_transcripts():
    clear_transcripts()
    return {"ok": True}


# ---------- 对话接口 ----------
@app.get("/api/llm/models")
def api_list_models():
    return {"models": get_llm().list_models()}


@app.post("/api/chat/stream")
async def api_chat_stream(body: dict = Body(...)):
    """
    流式 SSE 对话
    body: { "conversation_id": int|None, "message": str, "system_prompt": str|None }
    返回 SSE: data: {"text": "...", "done": false, "conversation_id": 1}
    """
    msg = body.get("message", "").strip()
    if not msg:
        raise HTTPException(400, "消息不能为空")

    system_prompt = body.get("system_prompt")
    cid = body.get("conversation_id")

    # 获取或创建会话
    if cid is None:
        cid = create_conversation()
    else:
        convs = list_conversations()
        if not any(c["id"] == cid for c in convs):
            raise HTTPException(404, "会话不存在")

    # 构建历史
    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    for m in get_messages(cid):
        history.append({"role": m["role"], "content": m["content"]})

    history.append({"role": "user", "content": msg})

    async def generate():
        full = ""
        for chunk in get_llm().chat(history, stream=True):
            full += chunk["text"]
            yield f"data: {json.dumps({'text': chunk['text'], 'done': chunk['done'], 'conversation_id': cid}, ensure_ascii=False)}\n\n"
            if chunk["done"]:
                # 保存到数据库
                add_message(cid, "user", msg)
                add_message(cid, "assistant", full)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/conversations")
def api_list_conversations():
    return list_conversations()


@app.post("/api/conversations")
def api_create_conversation(title: str = Form("新对话")):
    cid = create_conversation(title)
    return {"id": cid, "title": title}


@app.get("/api/conversations/{cid}/messages")
def api_get_messages(cid: int):
    return get_messages(cid)


@app.patch("/api/conversations/{cid}")
async def api_rename_conversation(cid: int, body: dict = Body(...)):
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(400, "标题不能为空")
    rename_conversation(cid, title)
    return {"ok": True}


@app.delete("/api/conversations/{cid}")
def api_delete_conversation(cid: int):
    delete_conversation(cid)
    return {"ok": True}


# ---------- 静态文件 ----------
# Web 前端页面
FRONTEND_HTML = None


def _get_frontend():
    global FRONTEND_HTML
    if FRONTEND_HTML is None:
        path = resource_path("frontend.html")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                FRONTEND_HTML = f.read()
    return FRONTEND_HTML


@app.get("/")
def index():
    html = _get_frontend()
    if html:
        return HTMLResponse(html)
    return FileResponse(resource_path("frontend.html"))


# 如果静态文件目录存在则挂载
STATIC_DIR = os.path.join(_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- 启动 ----------
def main():
    init_db()
    print("=" * 50)
    print("  AI 离线语音输入 & 对话 服务")
    print("  ASR 引擎: SenseVoice-Small ONNX")
    print(f"  LLM 引擎: {get_llm().model}")
    print("  Web 地址: http://127.0.0.1:8000")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()