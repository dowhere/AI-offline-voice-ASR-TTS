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
from llm_engine import get_engine as get_llm, LLMEngine, chat_stream, list_models_for, ping
from tts_engine import get_engine as get_tts, VOICES as TTS_VOICES
from db import init_db, add_transcript, list_transcripts, delete_transcript, clear_transcripts
from db import create_conversation, list_conversations, rename_conversation, delete_conversation
from db import add_message, get_messages
from db import (list_providers, add_provider, update_provider, delete_provider,
                set_default_provider, get_default_provider, get_provider)
from paths import APP_DIR, resource_path

_DIR = APP_DIR

app = FastAPI(title="AI 离线语音输入 & 对话", version="1.0.0")

# ---------- 引擎加载状态 ----------
ENGINE_STATUS = {
    "asr": "loading",   # loading | ready | error
    "llm": "loading",
    "tts": "loading",
    "asr_msg": "",
    "llm_msg": "",
    "tts_msg": "",
}


def _load_asr():
    try:
        get_engine()
        ENGINE_STATUS["asr"] = "ready"
        print("[启动] ASR 引擎就绪")
    except Exception as e:
        ENGINE_STATUS["asr"] = "error"
        ENGINE_STATUS["asr_msg"] = str(e)
        print(f"[启动] ASR 引擎加载失败: {e}")


def _load_llm():
    try:
        eng = get_llm()
        ENGINE_STATUS["llm"] = "ready"
        ENGINE_STATUS["llm_msg"] = getattr(eng, "model", "")
        print("[启动] LLM 引擎就绪")
    except Exception as e:
        ENGINE_STATUS["llm"] = "error"
        ENGINE_STATUS["llm_msg"] = str(e)
        print(f"[启动] LLM 引擎加载失败: {e}")


def _load_tts():
    try:
        get_tts()
        ENGINE_STATUS["tts"] = "ready"
        print("[启动] TTS 引擎就绪")
    except Exception as e:
        ENGINE_STATUS["tts"] = "error"
        ENGINE_STATUS["tts_msg"] = str(e)
        import traceback
        try:
            with open(os.path.join(APP_DIR, "tts_error.log"), "a", encoding="utf-8") as f:
                f.write("[warmup] " + traceback.format_exc() + "\n")
        except Exception:
            pass
        print(f"[启动] TTS 引擎加载失败: {e}")


# ---------- 启动时 ----------
@app.on_event("startup")
def startup():
    init_db()
    print("[启动] 数据库初始化完成，界面可立即使用")


_engines_started = False


def _start_engines():
    global _engines_started
    if _engines_started:
        return
    _engines_started = True
    print("[启动] 界面已就绪，开始后台加载模型引擎...")
    import threading
    for fn in (_load_asr, _load_llm, _load_tts):
        threading.Thread(target=fn, daemon=True).start()


@app.post("/api/engines/start")
def api_start_engines():
    """由 GUI 在窗口显示后调用, 触发模型加载 (界面未打开前不加载)"""
    _start_engines()
    return {"ok": True}


@app.get("/api/status")
def api_status():
    return ENGINE_STATUS


# ---------- ASR 接口 ----------
@app.post("/api/asr/upload")
async def asr_upload(file: UploadFile = File(...), language: str = Form("auto")):
    if ENGINE_STATUS["asr"] != "ready":
        raise HTTPException(503, "语音识别引擎正在加载中，请稍候…")
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
    if ENGINE_STATUS["asr"] != "ready":
        raise HTTPException(503, "语音识别引擎正在加载中，请稍候…")
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


# ---------- TTS 接口 ----------
@app.get("/api/tts/voices")
def api_tts_voices():
    return {"voices": [{"id": k, "name": v} for k, v in TTS_VOICES.items()]}


@app.post("/api/tts")
async def api_tts(body: dict = Body(...)):
    """文本合成语音, 返回 WAV 音频"""
    if ENGINE_STATUS["tts"] != "ready":
        raise HTTPException(503, "语音合成引擎正在加载中，请稍候…")
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "文本不能为空")
    voice = body.get("voice", "zf_xiaoxiao")
    try:
        speed = float(body.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    try:
        wav = get_tts().synthesize_wav(text, voice=voice, speed=speed)
        from fastapi.responses import Response
        return Response(content=wav, media_type="audio/wav")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            with open(os.path.join(APP_DIR, "tts_error.log"), "a", encoding="utf-8") as f:
                f.write(tb + "\n")
        except Exception:
            pass
        raise HTTPException(500, f"语音合成失败: {e}")



@app.post("/api/chat/stream")
async def api_chat_stream(body: dict = Body(...)):
    """
    流式 SSE 对话
    body: { conversation_id, message, system_prompt, provider_id, model }
    """
    msg = body.get("message", "").strip()
    if not msg:
        raise HTTPException(400, "消息不能为空")

    system_prompt = body.get("system_prompt")
    cid = body.get("conversation_id")

    # 选定 provider + model
    pid = body.get("provider_id")
    provider = get_provider(pid) if pid else get_default_provider()
    if not provider:
        raise HTTPException(400, "未配置任何大模型，请先在「模型管理」中添加")
    model = body.get("model") or provider.get("model") or ""
    if not model:
        raise HTTPException(400, "请先选择一个模型")

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
        try:
            for chunk in chat_stream(provider, model, history):
                full += chunk["text"]
                yield f"data: {json.dumps({'text': chunk['text'], 'done': chunk['done'], 'conversation_id': cid}, ensure_ascii=False)}\n\n"
                if chunk["done"]:
                    add_message(cid, "user", msg)
                    add_message(cid, "assistant", full)
        except Exception as e:
            yield f"data: {json.dumps({'text': f'[对话出错: {e}]', 'done': True, 'conversation_id': cid}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------- 模型管理 (Provider) ----------
@app.get("/api/providers")
def api_list_providers():
    # 不回传 api_key 明文, 只标记是否已配置
    out = []
    for p in list_providers():
        p = dict(p)
        p["has_key"] = bool(p.get("api_key"))
        p.pop("api_key", None)
        out.append(p)
    return out


@app.post("/api/providers")
async def api_add_provider(body: dict = Body(...)):
    name = (body.get("name") or "").strip()
    kind = body.get("kind", "openai")
    base_url = (body.get("base_url") or "").strip()
    if not name or not base_url:
        raise HTTPException(400, "名称和地址不能为空")
    pid = add_provider(name, kind, base_url,
                       api_key=body.get("api_key", ""),
                       model=body.get("model", ""),
                       is_default=1 if body.get("is_default") else 0)
    return {"id": pid}


@app.patch("/api/providers/{pid}")
async def api_update_provider(pid: int, body: dict = Body(...)):
    update_provider(pid, **body)
    return {"ok": True}


@app.delete("/api/providers/{pid}")
def api_delete_provider(pid: int):
    delete_provider(pid)
    return {"ok": True}


@app.post("/api/providers/{pid}/default")
def api_set_default(pid: int):
    set_default_provider(pid)
    return {"ok": True}


@app.get("/api/providers/{pid}/models")
def api_provider_models(pid: int):
    p = get_provider(pid)
    if not p:
        raise HTTPException(404, "配置不存在")
    try:
        return {"models": list_models_for(p)}
    except Exception as e:
        raise HTTPException(502, f"获取模型列表失败: {e}")


@app.post("/api/providers/test")
async def api_test_provider(body: dict = Body(...)):
    ok, m = ping({
        "kind": body.get("kind", "openai"),
        "base_url": (body.get("base_url") or "").strip(),
        "api_key": body.get("api_key", ""),
    })
    return {"ok": ok, "message": m}


@app.post("/api/providers/models_probe")
async def api_probe_models(body: dict = Body(...)):
    """测试连接并拉取模型列表 (供前端添加模型时使用)"""
    p = {
        "kind": body.get("kind", "openai"),
        "base_url": (body.get("base_url") or "").strip(),
        "api_key": body.get("api_key", ""),
    }
    try:
        models = list_models_for(p)
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "message": str(e)}


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


@app.get("/logo.png")
def logo():
    p = resource_path("VoiceRecorderApp.png")
    if os.path.exists(p):
        return FileResponse(p, media_type="image/png")
    raise HTTPException(404, "logo not found")


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