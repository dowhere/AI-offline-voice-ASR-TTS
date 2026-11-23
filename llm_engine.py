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

# (连接超时, 流式读取超时)
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 180


def _error_text(resp, limit=300):
    """从错误响应里提取可读信息"""
    try:
        body = resp.text or ""
    except Exception:
        body = ""
    body = " ".join(body.strip().split())
    return body[:limit] or f"HTTP {resp.status_code}"


def _check_response(resp, name):
    if resp.status_code >= 400:
        raise RuntimeError(f"{name} 返回 {resp.status_code}: {_error_text(resp)}")


# ---------- provider 探测 / 列模型 ----------
def list_models_for(provider):
    """列出某 provider 可用模型名"""
    kind = provider.get("kind", "ollama")
    base = (provider.get("base_url") or OLLAMA_BASE).rstrip("/")
    key = provider.get("api_key", "")
    if kind == "ollama":
        r = requests.get(f"{base}/api/tags", timeout=(CONNECT_TIMEOUT, 15))
        _check_response(r, "Ollama")
        return [m["name"] for m in r.json().get("models", [])]
    else:  # openai 兼容
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        r = requests.get(f"{base}/models", headers=headers, timeout=(CONNECT_TIMEOUT, 20))
        _check_response(r, "模型接口")
        data = r.json().get("data", [])
        return [m.get("id") for m in data if m.get("id")]


def ping(provider):
    """测试连通性, 返回 (ok, msg)"""
    try:
        models = list_models_for(provider)
        return True, f"连接成功，{len(models)} 个模型"
    except Exception as e:
        return False, f"连接失败: {e}"


# ---------- 对话 ----------
def _iter_ollama(base, model, messages):
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_predict": 2048, "temperature": 0.7},
    }
    resp = requests.post(
        f"{base}/api/chat", json=body, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    )
    done = False
    try:
        _check_response(resp, "Ollama")
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("error"):
                raise RuntimeError(f"Ollama 错误: {data['error']}")
            msg = data.get("message") or {}
            text = msg.get("content") or ""
            reasoning = msg.get("thinking") or ""
            is_done = bool(data.get("done"))
            if text or reasoning or is_done:
                yield {"text": text, "reasoning": reasoning, "done": is_done}
            if is_done:
                done = True
                break
    finally:
        resp.close()
    if not done:
        yield {"text": "", "reasoning": "", "done": True}


def _iter_openai(base, key, model, messages):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    resp = requests.post(
        f"{base}/chat/completions", json=body, headers=headers,
        stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    done = False
    try:
        _check_response(resp, "模型接口")
        for line in resp.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
            s = s.strip()
            if s.startswith("data:"):
                s = s[5:].strip()
            if not s:
                continue
            if s == "[DONE]":
                break
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("error"):
                err = data["error"]
                err = err.get("message") if isinstance(err, dict) else err
                raise RuntimeError(f"模型接口错误: {err}")
            choices = data.get("choices") or [{}]
            ch = choices[0] or {}
            delta = ch.get("delta") or {}
            text = delta.get("content") or ""
            # DeepSeek-R1 等推理模型: reasoning_content
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            finish = ch.get("finish_reason")
            if text or reasoning or finish:
                yield {"text": text, "reasoning": reasoning, "done": bool(finish)}
            if finish:
                done = True
                break
    finally:
        resp.close()
    if not done:
        yield {"text": "", "reasoning": "", "done": True}


def chat_stream(provider, model, messages):
    """统一流式对话入口, yield {'text':str,'reasoning':str,'done':bool}"""
    kind = provider.get("kind", "ollama")
    base = (provider.get("base_url") or OLLAMA_BASE).rstrip("/")
    key = provider.get("api_key", "")
    if kind == "ollama":
        yield from _iter_ollama(base, model, messages)
    else:
        yield from _iter_openai(base, key, model, messages)


# ---------- 兼容旧接口 (启动状态探测用) ----------
class LLMEngine:
    """薄封装: 默认连本地 Ollama, 供启动状态检测复用"""
    def __init__(self, base_url=OLLAMA_BASE, model=DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._check()

    def _check(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=(3, 5))
            _check_response(r, "Ollama")
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
        r = requests.get(f"{self.base_url}/api/tags", timeout=(3, 5))
        _check_response(r, "Ollama")
        return [m["name"] for m in r.json().get("models", [])]

    def chat(self, messages, stream=False, **kwargs):
        provider = {"kind": "ollama", "base_url": self.base_url, "api_key": ""}
        if stream:
            return chat_stream(provider, self.model, messages)
        full = ""
        for c in chat_stream(provider, self.model, messages):
            full += c.get("text", "")
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
