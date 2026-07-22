"""
LLM 对话引擎 — 支持多 provider
- ollama: 本地 Ollama (http://127.0.0.1:11434)
- openai: 任意 OpenAI 兼容接口 (URL + api-key + model)
provider 配置存于 SQLite (db.llm_providers), 对话时按需选用。
"""
import json
import requests

OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"


# ---------- provider 探测 / 列模型 ----------
def list_models_for(provider):
    """列出某 provider 可用模型名"""
    kind = provider.get("kind", "ollama")
    base = provider.get("base_url", "").rstrip("/")
    key = provider.get("api_key", "")
    if kind == "ollama":
        r = requests.get(f"{base}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    else:  # openai 兼容
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        r = requests.get(f"{base}/models", headers=headers, timeout=8)
        data = r.json().get("data", [])
        return [m["id"] for m in data]


def ping(provider):
    """测试连通性, 返回 (ok, msg)"""
    try:
        models = list_models_for(provider)
        return True, f"连接成功，{len(models)} 个模型"
    except Exception as e:
        return False, f"连接失败: {e}"


# ---------- 对话 ----------
def _messages_body(model, messages, stream, **opts):
    return {
        "model": model,
        "messages": messages,
        "stream": stream,
    }


def chat_stream(provider, model, messages):
    """统一流式对话入口, yield {'text':str,'done':bool}"""
    kind = provider.get("kind", "ollama")
    base = provider.get("base_url", "").rstrip("/")
    key = provider.get("api_key", "")

    if kind == "ollama":
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": 2048, "temperature": 0.7},
        }
        resp = requests.post(f"{base}/api/chat", json=body, stream=True, timeout=120)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            yield {"text": data.get("message", {}).get("content", ""),
                   "done": data.get("done", False)}
            if data.get("done"):
                break
    else:  # openai 兼容 (SSE)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {"model": model, "messages": messages, "stream": True,
                "temperature": 0.7, "max_tokens": 2048}
        resp = requests.post(f"{base}/chat/completions", json=body,
                             headers=headers, stream=True, timeout=120)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8") if isinstance(line, bytes) else line
            if s.startswith("data:"):
                s = s[5:].strip()
            if s == "[DONE]":
                yield {"text": "", "done": True}
                break
            try:
                data = json.loads(s)
            except Exception:
                continue
            delta = data.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content", "") or ""
            finish = data.get("choices", [{}])[0].get("finish_reason")
            yield {"text": piece, "done": bool(finish)}
            if finish:
                break


# ---------- 兼容旧接口 (启动状态探测用) ----------
class LLMEngine:
    """薄封装: 默认连本地 Ollama, 供启动状态检测复用"""
    def __init__(self, base_url=OLLAMA_BASE, model=DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._check()

    def _check(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            names = [m["name"] for m in r.json().get("models", [])]
            if self.model not in names:
                closest = [n for n in names if self.model.split(":")[0] in n]
                self.model = closest[0] if closest else (names[0] if names else self.model)
                print(f"[LLM] 使用模型: {self.model}")
            else:
                print(f"[LLM] 已连接, 使用模型: {self.model}")
        except Exception as e:
            print(f"[LLM] 警告: 无法连接 Ollama ({e})")

    def list_models(self):
        r = requests.get(f"{self.base_url}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]

    def chat(self, messages, stream=False, **kwargs):
        provider = {"kind": "ollama", "base_url": self.base_url, "api_key": ""}
        if stream:
            return chat_stream(provider, self.model, messages)
        full = ""
        for c in chat_stream(provider, self.model, messages):
            full += c["text"]
        return full


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = LLMEngine()
    return _engine


if __name__ == "__main__":
    eng = LLMEngine()
    print("可用模型:", eng.list_models())
