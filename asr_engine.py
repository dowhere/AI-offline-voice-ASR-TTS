"""
SenseVoice-Small ONNX 离线语音识别引擎
- 前端: kaldi fbank(80) -> LFR(m=7,n=6) -> 560维
- Encoder(fp16) -> CTC(fp16) -> greedy 解码 -> SentencePiece
支持: 中/英/粤/日/韩, 自动语种检测, 情感/事件标记
"""
import os
import re
import wave
import threading
import numpy as np
import soxr
import kaldi_native_fbank as knf
import sentencepiece as spm
import onnxruntime as ort

from paths import APP_DIR

MODEL_DIR = os.path.join(APP_DIR, "models", "Sensevoice-Small-ONNX")

LID = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
TEXTNORM = {"withitn": 14, "woitn": 15}

# 用于清理特殊标记, 提取纯文本
_TAG_RE = re.compile(r"<\|[^|]*\|>")
_EMO_MAP = {
    "HAPPY": "开心", "SAD": "难过", "ANGRY": "生气",
    "NEUTRAL": "平静", "FEARFUL": "害怕", "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶", "EMO_UNKNOWN": "",
}
_LANG_MAP = {"zh": "中文", "en": "英文", "yue": "粤语", "ja": "日语", "ko": "韩语"}


class ASREngine:
    def __init__(self, model_dir=MODEL_DIR, num_threads=4):
        enc_path = os.path.join(model_dir, "SenseVoice-Encoder.fp16.onnx")
        ctc_path = os.path.join(model_dir, "SenseVoice-CTC.fp16.onnx")
        tok_path = os.path.join(model_dir, "tokenizer.bpe.model")
        for p in (enc_path, ctc_path, tok_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"模型文件缺失: {p}")

        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(enc_path, sess_options=so, providers=providers)
        self.ctc = ort.InferenceSession(ctc_path, sess_options=so, providers=providers)
        self.sp = spm.SentencePieceProcessor()
        # Windows 非 ASCII 路径下 sentencepiece 无法直接按路径加载, 改用字节流
        with open(tok_path, "rb") as f:
            self.sp.load_from_serialized_proto(f.read())
        self.vocab_size = self.sp.get_piece_size()
        # onnxruntime 本身线程安全, 但这里串行化可避免并发请求争抢 CPU 与内存峰值
        self._lock = threading.Lock()

    # ---------- 前端 ----------
    @staticmethod
    def _read_wav(path):
        with wave.open(path, "rb") as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(n)
        if sw == 1:                       # 8-bit WAV 是无符号
            audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sw == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 3:                     # 24-bit
            b = np.frombuffer(raw, dtype=np.uint8)
            b = b[: (len(b) // 3) * 3].reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v >= (1 << 23), v - (1 << 24), v)
            audio = v.astype(np.float32) / float(1 << 23)
        elif sw == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"不支持的 WAV 位深: {sw * 8} bit")
        if ch > 1:
            audio = audio.reshape(-1, ch).mean(axis=1)
        if sr != 16000:
            audio = soxr.resample(audio, sr, 16000)
        return np.ascontiguousarray(audio, dtype=np.float32)

    @staticmethod
    def _fbank(audio):
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.dither = 0.0
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.frame_shift_ms = 10.0
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        fb = knf.OnlineFbank(opts)
        fb.accept_waveform(16000, (audio * 32768).tolist())
        fb.input_finished()
        return np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], dtype=np.float32)

    @staticmethod
    def _lfr(feat, m=7, n=6):
        if feat.shape[0] == 0:
            return np.zeros((0, feat.shape[1] * m), dtype=np.float32)
        T = feat.shape[0]
        T_lfr = int(np.ceil(T / n))
        left = np.tile(feat[0], ((m - 1) // 2, 1))
        padded = np.vstack([left, feat])
        Tp = padded.shape[0]
        out = []
        for i in range(T_lfr):
            s = i * n
            if s + m <= Tp:
                out.append(padded[s:s + m].reshape(-1))
            else:
                frame = padded[s:].reshape(-1)
                pad = m - (Tp - s)
                frame = np.concatenate([frame] + [padded[-1]] * pad)
                out.append(frame)
        return np.array(out, dtype=np.float32)

    # ---------- 推理 ----------
    def _infer(self, audio, language="auto", use_itn=True):
        if audio is None or audio.size < 160:   # 少于 10ms 视为无语音
            return ""
        feat = self._fbank(audio)
        if feat.shape[0] == 0:
            return ""
        lfr = self._lfr(feat)
        T = lfr.shape[0]
        speech = lfr[None].astype(np.float16)
        mask = np.ones((1, T), dtype=np.float16)
        lang_id = LID.get(language, 0)
        norm_id = TEXTNORM["withitn"] if use_itn else TEXTNORM["woitn"]
        prompt = np.array([[lang_id, 1, 2, norm_id]], dtype=np.int64)

        with self._lock:
            enc_out = self.enc.run(
                None, {"speech_feat": speech, "mask": mask, "prompt_ids": prompt}
            )[0]
            _, topk_idx = self.ctc.run(None, {"enc_out": enc_out.astype(np.float16)})
        top1 = topk_idx[0, :, 0].tolist()

        ids, prev = [], -1
        for t in top1:
            if t != prev and t != 0:
                ids.append(t)
            prev = t
        return self.sp.decode(ids)

    @staticmethod
    def _parse(raw):
        """从带标记的原始输出中拆出: 纯文本 / 语种 / 情感"""
        tags = _TAG_RE.findall(raw)
        lang, emo = "", ""
        for t in tags:
            inner = t[2:-2]
            if inner in _LANG_MAP:
                lang = _LANG_MAP[inner]
            up = inner.upper()
            if up in _EMO_MAP:
                emo = _EMO_MAP[up]
        text = _TAG_RE.sub("", raw).strip()
        return {"text": text, "language": lang, "emotion": emo, "raw": raw}

    def transcribe_file(self, path, language="auto", use_itn=True):
        audio = self._read_wav(path)
        raw = self._infer(audio, language, use_itn)
        r = self._parse(raw)
        r["duration"] = round(len(audio) / 16000, 2)
        return r

    def transcribe_pcm16(self, pcm_bytes, sample_rate, language="auto", use_itn=True):
        """直接处理原始 PCM16 单声道字节流"""
        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            sample_rate = 16000
        if sample_rate <= 0:
            sample_rate = 16000
        if len(pcm_bytes) % 2:                 # 丢弃不完整的半采样
            pcm_bytes = pcm_bytes[:-1]
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return {"text": "", "language": "", "emotion": "", "raw": "", "duration": 0.0}
        if sample_rate != 16000:
            audio = soxr.resample(audio, sample_rate, 16000)
        raw = self._infer(np.ascontiguousarray(audio, dtype=np.float32), language, use_itn)
        r = self._parse(raw)
        r["duration"] = round(len(audio) / 16000, 2)
        return r


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = ASREngine()
    return _engine


if __name__ == "__main__":
    eng = ASREngine()
    for w in ["test_zh.wav", "test_en.wav"]:
        if os.path.exists(w):
            print(w, "->", eng.transcribe_file(w))
