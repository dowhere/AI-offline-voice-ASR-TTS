"""
Kokoro TTS 离线语音合成引擎
- 模型: KokoroTTS (v1.0, 24kHz)
- 中文 G2P: misaki[zh] (jieba)  英文 G2P: espeak-ng
- 多说话人: 中文 zf_*/zm_*, 英文 af_*/am_*/bf_*/bm_* 等
"""
import os
import re
import threading
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _setup_espeak():
    """
    修正 espeak-ng 数据/库路径。两个坑:
    1. espeak-ng.dll 硬编码了构建机路径 -> 用 ESPEAK_DATA_PATH 覆盖
    2. dll 的 C 层无法读取含非 ASCII (如中文) 的路径 -> 若当前数据路径含非 ASCII,
       复制一份到 ASCII 安全的临时目录再指向它
    仅英文合成依赖 espeak; 中文走 misaki/jieba, 不受影响。
    """
    try:
        import espeakng_loader
        lib = espeakng_loader.get_library_path()
        data = espeakng_loader.get_data_path()  # .../espeak-ng-data
        data_parent = os.path.dirname(data)

        # 路径含非 ASCII 则复制到临时 ASCII 目录
        if not data_parent.isascii():
            import tempfile
            import shutil
            safe_parent = os.path.join(tempfile.gettempdir(), "kokoro_espeak")
            safe_data = os.path.join(safe_parent, "espeak-ng-data")
            if not os.path.exists(os.path.join(safe_data, "phontab")):
                os.makedirs(safe_parent, exist_ok=True)
                shutil.copytree(data, safe_data, dirs_exist_ok=True)
            data_parent = safe_parent
            data = safe_data

        os.environ["ESPEAK_DATA_PATH"] = data_parent
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        if hasattr(EspeakWrapper, "set_library"):
            EspeakWrapper.set_library(lib)
        if hasattr(EspeakWrapper, "set_data_path"):
            EspeakWrapper.set_data_path(data)
    except Exception:
        pass


_setup_espeak_done = False


def _ensure_espeak():
    global _setup_espeak_done
    if not _setup_espeak_done:
        _setup_espeak()
        _setup_espeak_done = True


from paths import APP_DIR

MODEL_DIR = os.path.join(APP_DIR, "models", "KokoroTTS")
SAMPLE_RATE = 24000
MAX_CHARS = 2000          # 单次合成上限, 避免长文本占用过久

# 常用中文/英文发音人 (name -> 说明)
VOICES = {
    "zf_xiaoxiao": "小晓 (女·中文)",
    "zf_xiaoni": "小妮 (女·中文)",
    "zf_xiaobei": "小北 (女·中文)",
    "zf_xiaoyi": "小艺 (女·中文)",
    "zm_yunxi": "云希 (男·中文)",
    "zm_yunjian": "云健 (男·中文)",
    "zm_yunyang": "云扬 (男·中文)",
    "zm_yunxia": "云夏 (男·中文)",
    "af_heart": "Heart (女·英文)",
    "af_bella": "Bella (女·英文)",
    "af_nicole": "Nicole (女·英文)",
    "af_sarah": "Sarah (女·英文)",
    "am_michael": "Michael (男·英文)",
    "am_adam": "Adam (男·英文)",
    "am_onyx": "Onyx (男·英文)",
    "bf_emma": "Emma (女·英式)",
    "bm_george": "George (男·英式)",
}

# 中文字符检测
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")


def clean_for_speech(text):
    """去掉 Markdown / 思考过程等不适合朗读的内容"""
    text = text or ""
    text = re.sub(r"<think>[\s\S]*?</think>", " ", text, flags=re.I)
    text = re.sub(r"<\|[^|]*\|>", " ", text)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = text.replace("`", "").replace("~~", "")
    text = re.sub(r"https?://\S+", "链接", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate(text, limit=MAX_CHARS):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # 尽量在句子边界截断
    idx = max(cut.rfind(c) for c in "。！？!?.\n")
    if idx >= limit // 2:
        cut = cut[: idx + 1]
    return cut.strip()


class TTSEngine:
    def __init__(self, model_dir=MODEL_DIR):
        cfg = os.path.join(model_dir, "config.json")
        ckpt = os.path.join(model_dir, "kokoro-v1_0.pth")
        for p in (cfg, ckpt):
            if not os.path.exists(p):
                raise FileNotFoundError(f"KokoroTTS 模型文件缺失: {p}")
        self.model_dir = model_dir
        self.voices_dir = os.path.join(model_dir, "voices")

        _ensure_espeak()
        from kokoro import KModel, KPipeline
        self._KPipeline = KPipeline
        self.model = KModel(config=cfg, model=ckpt).eval()
        # 中/英两套 pipeline (共享同一模型权重, 只是 G2P 前端不同)
        self._pipes = {}
        self._voice_cache = {}
        # torch/Kokoro 推理非线程安全, 串行化请求
        self._lock = threading.Lock()

    def _pipe(self, lang_code):
        if lang_code not in self._pipes:
            self._pipes[lang_code] = self._KPipeline(
                lang_code=lang_code, model=self.model, repo_id="hexgrad/Kokoro-82M"
            )
        return self._pipes[lang_code]

    def _load_voice(self, name):
        if name not in self._voice_cache:
            import torch
            path = os.path.join(self.voices_dir, f"{name}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"发音人不存在: {name}")
            self._voice_cache[name] = torch.load(path, weights_only=True)
        return self._voice_cache[name]

    @staticmethod
    def _lang_of(voice_name):
        # Kokoro 语言码: a=美英 b=英英 e=西 f=法 h=印地 i=意 j=日 p=葡 z=中
        first = (voice_name or "a")[0].lower()
        return first if first in "abefhijpz" else "a"

    def synthesize(self, text, voice="zf_xiaoxiao", speed=1.0):
        """合成 -> 返回 float32 numpy 波形 (24kHz 单声道)"""
        text = clean_for_speech(text)
        if not text:
            return np.zeros(0, dtype=np.float32)
        text = _truncate(text)
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.5, min(2.0, speed))

        pipe = self._pipe(self._lang_of(voice))
        chunks = []
        import torch
        with self._lock, torch.inference_mode():
            vt = self._load_voice(voice)
            for r in pipe(text, voice=vt, speed=speed):
                if r.audio is not None:
                    chunks.append(r.audio.detach().cpu().numpy())
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    def synthesize_wav(self, text, voice="zf_xiaoxiao", speed=1.0):
        """合成 -> 返回 WAV 字节 (可直接写文件或走 HTTP)"""
        audio = self.synthesize(text, voice, speed)
        return pcm_to_wav(audio, SAMPLE_RATE)


def pcm_to_wav(audio_f32, sample_rate=SAMPLE_RATE):
    import io
    import wave
    pcm = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    w.writeframes(pcm.tobytes())
    w.close()
    return buf.getvalue()


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine


if __name__ == "__main__":
    eng = TTSEngine()
    wav = eng.synthesize_wav("你好，这是Kokoro语音合成引擎测试。", voice="zf_xiaoxiao")
    with open("tts_engine_test.wav", "wb") as f:
        f.write(wav)
    print("中文合成 OK, bytes:", len(wav))
