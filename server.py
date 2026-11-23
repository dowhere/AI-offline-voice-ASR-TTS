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
import re
import json
import uuid
import asyncio
import tempfile
import threading
import subprocess

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request
from fastapi.responses import StreamingResponse, FileResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import uvicorn

from asr_engine import get_engine
from llm_engine import get_engine as get_llm, chat_stream, list_models_for, ping
from tts_engine import get_engine as get_tts, VOICES as TTS_VOICES
from db import init_db, add_transcript, list_transcripts, delete_transcript, clear_transcripts
from db import create_conversation, list_conversations, rename_conversation, delete_conversation
from db import add_message, get_messages, get_conversation, count_messages
from db import (list_providers, add_provider, update_provider, delete_provider,
                set_default_provider, get_default_provider, get_provider)
from paths import APP_DIR, resource_path

_DIR = APP_DIR

# 限制一次性输入体积, 避免内存/耗时失控
MAX_UPLOAD_BYTES = 100 * 1024 * 1024      # 100MB
MAX_PCM_BYTES = 50 * 1024 * 1024         # 50MB (约 27 分钟 16k 单声道)

# 未指定系统提示时的默认人设 (更适合口语化 / 语音朗读)
DEFAULT_SYSTEM_PROMPT = (
    "你是一个中文语音助手。请用简洁、口语化的方式回答，"
    "便于语音朗读；除非用户明确要求，不要输出大段 Markdown 表格或代码块。"
)

app = FastAPI(title="AI 离线语音输入 & 对话", version="1.1.0")

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


def _save_transcript(result, source):
    """保存识别结果; 空文本不入库。返回是否已保存。"""
    text = (result.get("text") or "").strip()
    if not text:
        return False
    result["text"] = text
    result["id"] = add_transcript(
        text=text,
        language=result.get("language", ""),
        emotion=result.get("emotion", ""),
        duration=result.get("duration", 0.0),
        source=source,
    )
    return True


def _convert_to_wav16k(src, dst):
    """用 ffmpeg 统一转为 16kHz/16bit/单声道 WAV。
    找不到 ffmpeg 或转码失败时, 若源文件本身就是 wav 则交给引擎自行解码。
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1",
             "-sample_fmt", "s16", dst],
            capture_output=True, timeout=60,
        )
    except FileNotFoundError:
        if src.lower().endswith(".wav"):
            return src
        raise HTTPException(500, "未找到 ffmpeg，无法转码该音频格式，请改用 WAV 文件")
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "音频转码超时")

    if proc.returncode != 0 or not os.path.exists(dst):
        if src.lower().endswith(".wav"):
            # ffmpeg 不可用/失败, 但引擎本身支持常见 WAV
            return src
        tail = (proc.stderr or b"").decode("utf-8", "ignore").strip().splitlines()
        detail = tail[-1][:300] if tail else f"退出码 {proc.returncode}"
        raise HTTPException(500, f"音频转码失败: {detail}")
    return dst


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
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "文件过大（上限 100MB）")

    # 保存临时文件, 用 ffmpeg 统一转为 16kHz 16bit WAV
    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext):
        ext = ".wav"
    tmp_in = os.path.join(tempfile.gettempdir(), f"asr_{uuid.uuid4().hex}{ext}")
    tmp_out = os.path.join(tempfile.gettempdir(), f"asr_{uuid.uuid4().hex}.wav")
    try:
        with open(tmp_in, "wb") as f:
            f.write(content)
        wav_path = await run_in_threadpool(_convert_to_wav16k, tmp_in, tmp_out)
        result = await run_in_threadpool(
            lambda: get_engine().transcribe_file(wav_path, language=language)
        )
        result["source"] = "upload"
        result["saved"] = _save_transcript(result, "upload")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"识别失败: {e}")
    finally:
        for p in (tmp_in, tmp_out):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


@app.post("/api/asr/record")
async def asr_record(request: Request, sample_rate: int = 16000, language: str = "auto"):
    """接收浏览器录音的 PCM16 原始字节流 (application/octet-stream)"""
    if ENGINE_STATUS["asr"] != "ready":
        raise HTTPException(503, "语音识别引擎正在加载中，请稍候…")
    if not (8000 <= sample_rate <= 96000):
        raise HTTPException(400, "sample_rate 不合法")
    data = await request.body()
    if not data:
        raise HTTPException(400, "空数据")
    if len(data) > MAX_PCM_BYTES:
        raise HTTPException(413, "录音过长（上限约 27 分钟）")
    min_bytes = max(320, int(sample_rate * 0.2) * 2)   # 至少 200ms, 太短易误识别
    if len(data) < min_bytes:
        raise HTTPException(400, "录音太短，请至少说 0.2 秒")
    try:
        result = await run_in_threadpool(
            lambda: get_engine().transcribe_pcm16(data, sample_rate, language=language)
        )
        result["source"] = "record"
        result["saved"] = _save_transcript(result, "record")
        return result
    except Exception as e:
        raise HTTPException(500, f"识别失败: {e}")


@app.get("/api/transcripts")
def api_list_transcripts(limit: int = 200):
    limit = max(1, min(int(limit), 1000))
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
_VOICE_RE = re.compile(r"^[a-z]{1,3}_[a-z0-9]+$")


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
    voice = (body.get("voice") or "zf_xiaoxiao").strip()
    if not _VOICE_RE.match(voice):
        raise HTTPException(400, "发音人参数不合法")
    try:
        speed = float(body.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    try:
        wav = await run_in_threadpool(
            lambda: get_tts().synthesize_wav(text, voice=voice, speed=speed)
        )
        if len(wav) <= 44:          # 只有 WAV 头, 没有音频数据
            raise HTTPException(500, "语音合成结果为空，请换一段文本试试")
        return Response(content=wav, media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            with open(os.path.join(APP_DIR, "tts_error.log"), "a", encoding="utf-8") as f:
                f.write(tb + "\n")
        except Exception:
            pass
        raise HTTPException(500, f"语音合成失败: {e}")


def _strip_think(text):
    return re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.I).strip()


def _title_from(text, limit=24):
    text = " ".join((text or "").split())
    if not text:
        return "新对话"
    return text[:limit] + ("…" if len(text) > limit else "")


async def _chat_stream_async(provider, model, history):
    """把阻塞式 requests 流放到后台线程, 避免阻塞事件循环导致状态轮询/其它接口卡住"""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def worker():
        try:
            for chunk in chat_stream(provider, model, history):
                if cancel.is_set():
                    return
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
        except Exception as e:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
            except RuntimeError:
                pass
        finally:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
            except RuntimeError:
                pass

    threading.Thread(target=worker, daemon=True, name="chat-stream").start()
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "chunk":
                yield payload
            elif kind == "error":
                raise RuntimeError(payload)
            else:
                return
    finally:
        cancel.set()


@app.post("/api/chat/stream")
async def api_chat_stream(body: dict = Body(...)):
    """
    流式 SSE 对话
    body: { conversation_id, message, system_prompt, provider_id, model }
    """
    msg = (body.get("message") or "").strip()
    if not msg:
        raise HTTPException(400, "消息不能为空")

    system_prompt = (body.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
    cid = body.get("conversation_id")

    # 选定 provider + model
    pid = body.get("provider_id")
    provider = get_provider(pid) if pid else get_default_provider()
    if not provider:
        raise HTTPException(400, "未配置任何大模型，请先在「模型管理」中添加")
    model = (body.get("model") or provider.get("model") or "").strip()
    if not model:
        raise HTTPException(400, "请先选择一个模型")

    # 获取或创建会话; 新会话用首条消息自动命名
    if cid is None:
        cid = create_conversation(_title_from(msg))
    else:
        conv = get_conversation(cid)
        if not conv:
            raise HTTPException(404, "会话不存在")
        if count_messages(cid) == 0 and (conv.get("title") or "") in ("", "新对话"):
            rename_conversation(cid, _title_from(msg))

    # 构建历史 (思考过程不回传给模型)
    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    for m in get_messages(cid):
        history.append({"role": m["role"], "content": _strip_think(m["content"])})
    history.append({"role": "user", "content": msg})

    async def generate():
        full, thought = "", ""
        try:
            async for chunk in _chat_stream_async(provider, model, history):
                piece = chunk.get("text") or ""
                reasoning = chunk.get("reasoning") or ""
                if reasoning:
                    thought += reasoning
                full += piece
                payload = {
                    "text": piece,
                    "done": bool(chunk.get("done")),
                    "conversation_id": cid,
                }
                if reasoning:
                    payload["reasoning"] = reasoning
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            try:
                yield f"data: {json.dumps({'text': '', 'error': str(e), 'done': True, 'conversation_id': cid}, ensure_ascii=False)}\n\n"
            except Exception:
                pass
        finally:
            # 无论正常结束、报错还是客户端断开, 都落库已生成的部分
            try:
                add_message(cid, "user", msg)
                content = (f"<think>{thought}</think>\n" if thought else "") + full
                if content.strip():
                    add_message(cid, "assistant", content)
            except Exception as e:
                print(f"[对话] 保存消息失败: {e}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------- 模型管理 (Provider) ----------
def _public_provider(p):
    p = dict(p)
    p["has_key"] = bool(p.get("api_key"))
    p.pop("api_key", None)
    return p


def _clean_base_url(url):
    url = (url or "").strip().rstrip("/")
    if not re.match(r"^https?://", url, re.I):
        raise HTTPException(400, "接口地址需以 http:// 或 https:// 开头")
    return url


@app.get("/api/providers")
def api_list_providers():
    # 不回传 api_key 明文, 只标记是否已配置
    return [_public_provider(p) for p in list_providers()]


@app.get("/api/providers/{pid}")
def api_get_provider(pid: int):
    p = get_provider(pid)
    if not p:
        raise HTTPException(404, "配置不存在")
    return _public_provider(p)


@app.post("/api/providers")
async def api_add_provider(body: dict = Body(...)):
    name = (body.get("name") or "").strip()
    kind = body.get("kind", "openai")
    if kind not in ("ollama", "openai"):
        raise HTTPException(400, "不支持的类型")
    base_url = _clean_base_url(body.get("base_url"))
    if not name:
        raise HTTPException(400, "名称不能为空")
    pid = add_provider(name, kind, base_url,
                       api_key=body.get("api_key") or "",
                       model=(body.get("model") or "").strip(),
                       is_default=1 if body.get("is_default") else 0)
    return {"id": pid}


@app.patch("/api/providers/{pid}")
async def api_update_provider(pid: int, body: dict = Body(...)):
    if not get_provider(pid):
        raise HTTPException(404, "配置不存在")
    fields = {}
    for key in ("name", "base_url", "model", "api_key", "is_default", "kind"):
        if key in body:
            fields[key] = body[key]
    if "name" in fields:
        fields["name"] = (fields["name"] or "").strip()
        if not fields["name"]:
            raise HTTPException(400, "名称不能为空")
    if "kind" in fields and fields["kind"] not in ("ollama", "openai"):
        raise HTTPException(400, "不支持的类型")
    if "base_url" in fields:
        fields["base_url"] = _clean_base_url(fields["base_url"])
    if "model" in fields:
        fields["model"] = (fields["model"] or "").strip()
    # 编辑时留空表示不修改已保存的 Key, 避免误清空
    if not fields.get("api_key"):
        fields.pop("api_key", None)
    update_provider(pid, **fields)
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
def api_test_provider(body: dict = Body(...)):
    kind = body.get("kind", "openai")
    if kind not in ("ollama", "openai"):
        raise HTTPException(400, "不支持的类型")
    ok, m = ping({
        "kind": kind,
        "base_url": _clean_base_url(body.get("base_url")),
        "api_key": body.get("api_key") or "",
    })
    return {"ok": ok, "message": m}


@app.post("/api/providers/models_probe")
def api_probe_models(body: dict = Body(...)):
    """测试连接并拉取模型列表 (供前端添加模型时使用)"""
    kind = body.get("kind", "openai")
    if kind not in ("ollama", "openai"):
        return {"ok": False, "message": "不支持的类型"}
    p = {
        "kind": kind,
        "base_url": (body.get("base_url") or "").strip().rstrip("/"),
        "api_key": body.get("api_key") or "",
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
    if not get_conversation(cid):
        raise HTTPException(404, "会话不存在")
    return get_messages(cid)


@app.patch("/api/conversations/{cid}")
async def api_rename_conversation(cid: int, body: dict = Body(...)):
    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(400, "标题不能为空")
    if not get_conversation(cid):
        raise HTTPException(404, "会话不存在")
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