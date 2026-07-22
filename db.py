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
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
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
        """
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


if __name__ == "__main__":
    init_db()
    print("DB 初始化完成:", DB_PATH)
