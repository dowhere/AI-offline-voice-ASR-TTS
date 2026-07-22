"""
本地 Ollama LLM 对话引擎
支持流式 SSE 输出, 支持对话历史管理
"""
import json
import requests
import time

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


class LLMEngine:
    def __init__(self, base_url=OLLAMA_BASE, model=DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # 验证可用性
        self._check()

    def _check(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            models = r.json().get("models", [])
            names = [m["name"] for m in models]
            if self.model not in names:
                closest = [n for n in names if self.model.split(":")[0] in n]
                if closest:
                    self.model = closest[0]
                    print(f"[LLM] 切换模型: {closest[0]}")
                else:
                    print(f"[LLM] 警告: {self.model} 未找到, 可用模型: {names}")
            else:
                print(f"[LLM] 已连接, 使用模型: {self.model}")
        except Exception as e:
            print(f"[LLM] 警告: 无法连接 Ollama ({e}), 请确保 Ollama 在运行")
            print(f"       启动: ollama serve")

    def list_models(self):
        r = requests.get(f"{self.base_url}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]

    def chat(self, messages, stream=False, **kwargs):
        """
        messages: [{"role":"user"/"assistant"/"system", "content":"..."}]
        stream=True 返回生成器, 每次 yield {"text": str, "done": bool}
        """
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"num_predict": 2048, "temperature": 0.7, **kwargs},
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=body,
            stream=stream,
            timeout=60,
        )
        if stream:
            return self._stream(resp)
        else:
            return resp.json()["message"]["content"]

    @staticmethod
    def _stream(resp):
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            yield {"text": data.get("message", {}).get("content", ""), "done": data.get("done", False)}
            if data.get("done"):
                break

    def chat_with_history(self, history, user_msg, system_prompt=None):
        """快捷方法: 追加用户消息, 流式回复, 返回完整回复并更新 history"""
        if system_prompt:
            history = [m for m in history if m["role"] != "system"]
            history.insert(0, {"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": user_msg})
        full = ""
        for chunk in self.chat(history, stream=True):
            full += chunk["text"]
        history.append({"role": "assistant", "content": full})
        return full, history


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = LLMEngine()
    return _engine


if __name__ == "__main__":
    eng = LLMEngine()
    print("可用模型:", eng.list_models())
    resp, _ = eng.chat_with_history([], "用一句话介绍自己")
    print("回复:", resp)