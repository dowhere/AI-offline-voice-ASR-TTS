"""
SQLite 存储层
- transcripts: 语音识别记录
- conversations: 对话会话
- messages: 对话消息
"""
import sqlite3
import os
import time
import threading

from paths import APP_DIR

DB_PATH = os.path.join(APP_DIR, "data.db")

_local = threading.local()


def _conn():
    if not hasattr(_local, "conn"):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        # WAL + busy_timeout: 识别/对话/前端轮询并发访问时避免 database is locked
        for pragma in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA busy_timeout=10000",
            "PRAGMA foreign_keys=ON",
        ):
            try:
                conn.execute(pragma)
            except sqlite3.Error:
                pass
        _local.conn = conn
    return _local.conn


def init_db():
    c = _conn()
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            language TEXT,
            emotion TEXT,
            duration REAL,
            source TEXT,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        );
        CREATE TABLE IF NOT EXISTS llm_providers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,            -- ollama | openai
            base_url TEXT NOT NULL,
            api_key TEXT DEFAULT '',
            model TEXT DEFAULT '',
            is_default INTEGER DEFAULT 0,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_transcripts_created ON transcripts(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
        """
    )
    c.commit()
    _seed_default_provider()


def _seed_default_provider():
    """首次运行时植入本地 Ollama 默认 provider"""
    c = _conn()
    n = c.execute("SELECT COUNT(*) AS n FROM llm_providers").fetchone()["n"]
    if n == 0:
        c.execute(
            "INSERT INTO llm_providers(name,kind,base_url,api_key,model,is_default,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            ("本地 Ollama", "ollama", "http://127.0.0.1:11434", "", "", 1, int(time.time())),
        )
        c.commit()


# ---------- 识别记录 ----------
def add_transcript(text, language="", emotion="", duration=0.0, source="upload"):
    c = _conn()
    cur = c.execute(
        "INSERT INTO transcripts(text,language,emotion,duration,source,created_at) VALUES(?,?,?,?,?,?)",
        (text, language, emotion, duration, source, int(time.time())),
    )
    c.commit()
    return cur.lastrowid


def list_transcripts(limit=200):
    c = _conn()
    rows = c.execute(
        "SELECT * FROM transcripts ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_transcript(tid):
    c = _conn()
    c.execute("DELETE FROM transcripts WHERE id=?", (tid,))
    c.commit()


def clear_transcripts():
    c = _conn()
    c.execute("DELETE FROM transcripts")
    c.commit()


# ---------- 会话 ----------
def create_conversation(title="新对话"):
    c = _conn()
    cur = c.execute(
        "INSERT INTO conversations(title,created_at) VALUES(?,?)",
        (title, int(time.time())),
    )
    c.commit()
    return cur.lastrowid


def list_conversations():
    c = _conn()
    rows = c.execute("SELECT * FROM conversations ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def get_conversation(cid):
    c = _conn()
    row = c.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    return dict(row) if row else None


def count_messages(cid):
    c = _conn()
    row = c.execute("SELECT COUNT(*) AS n FROM messages WHERE conversation_id=?", (cid,)).fetchone()
    return int(row["n"]) if row else 0


def rename_conversation(cid, title):
    c = _conn()
    c.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))
    c.commit()


def delete_conversation(cid):
    c = _conn()
    c.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    c.execute("DELETE FROM conversations WHERE id=?", (cid,))
    c.commit()


def add_message(cid, role, content):
    c = _conn()
    cur = c.execute(
        "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
        (cid, role, content, int(time.time())),
    )
    c.commit()
    return cur.lastrowid


def get_messages(cid):
    c = _conn()
    rows = c.execute(
        "SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY id ASC",
        (cid,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------- LLM Provider (模型配置) ----------
def list_providers():
    c = _conn()
    rows = c.execute("SELECT * FROM llm_providers ORDER BY id ASC").fetchall()
    return [dict(r) for r in rows]


def add_provider(name, kind, base_url, api_key="", model="", is_default=0):
    c = _conn()
    if is_default:
        c.execute("UPDATE llm_providers SET is_default=0")
    cur = c.execute(
        "INSERT INTO llm_providers(name,kind,base_url,api_key,model,is_default,created_at) "
        "VALUES(?,?,?,?,?,?,?)",
        (name, kind, base_url.rstrip("/"), api_key, model, int(bool(is_default)), int(time.time())),
    )
    c.commit()
    return cur.lastrowid


def update_provider(pid, **fields):
    allowed = {"name", "kind", "base_url", "api_key", "model", "is_default"}
    sets, vals = [], []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "base_url" and isinstance(v, str):
            v = v.strip().rstrip("/")
        if k == "is_default":
            v = int(bool(v))
        sets.append(f"{k}=?")
        vals.append(v)
    if not sets:
        return
    c = _conn()
    if fields.get("is_default"):
        c.execute("UPDATE llm_providers SET is_default=0")
    vals.append(pid)
    c.execute(f"UPDATE llm_providers SET {','.join(sets)} WHERE id=?", vals)
    c.commit()


def delete_provider(pid):
    c = _conn()
    row = c.execute("SELECT is_default FROM llm_providers WHERE id=?", (pid,)).fetchone()
    c.execute("DELETE FROM llm_providers WHERE id=?", (pid,))
    # 若删的是默认项, 把第一个设为默认
    if row and row["is_default"]:
        first = c.execute("SELECT id FROM llm_providers ORDER BY id ASC LIMIT 1").fetchone()
        if first:
            c.execute("UPDATE llm_providers SET is_default=1 WHERE id=?", (first["id"],))
    c.commit()


def set_default_provider(pid):
    c = _conn()
    c.execute("UPDATE llm_providers SET is_default=0")
    c.execute("UPDATE llm_providers SET is_default=1 WHERE id=?", (pid,))
    c.commit()


def get_default_provider():
    c = _conn()
    row = c.execute("SELECT * FROM llm_providers WHERE is_default=1 LIMIT 1").fetchone()
    if not row:
        row = c.execute("SELECT * FROM llm_providers ORDER BY id ASC LIMIT 1").fetchone()
    return dict(row) if row else None


def get_provider(pid):
    c = _conn()
    row = c.execute("SELECT * FROM llm_providers WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None


if __name__ == "__main__":
    init_db()
    print("DB 初始化完成:", DB_PATH)
