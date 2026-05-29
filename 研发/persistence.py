# -*- coding: utf-8 -*-
"""
SQLite 持久化存储模块（Persistence Layer）

功能概述：
- 使用 SQLite 存储用户信息与对话历史
- 为 RAG / 多角色对话系统提供长期记忆
- 支持管理员统计与查询
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SQLiteStore:
    """
    SQLite 数据存储封装类

    职责：
    - 用户管理（注册、查询、更新）
    - 对话历史存储与检索
    - 管理员统计与审计
    """

    def __init__(self, db_path: str):
        """
        初始化数据库连接与表结构

        :param db_path: SQLite 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connection(self):
        """
        SQLite 连接上下文管理器

        - 自动提交事务
        - 自动关闭连接
        - 使用 sqlite3.Row 作为行工厂，便于字典式访问
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        """
        初始化数据库表结构

        - users：用户基本信息
        - messages：用户与角色的对话记录
        - idx_messages_user_role_created：联合索引，优化查询性能
        """
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    preferred_role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    role_id TEXT NOT NULL,
                    message_role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(username) REFERENCES users(username)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_user_role_created "
                "ON messages(username, role_id, created_at)"
            )

    @staticmethod
    def _now() -> str:
        """
        获取当前 UTC 时间（ISO 8601 格式，精确到秒）
        """
        return datetime.utcnow().isoformat(timespec="seconds")

    # =========================
    # 用户相关操作
    # =========================

    def create_user(
        self,
        username: str,
        email: str,
        hashed_password: str,
        preferred_role: str,
    ) -> None:
        """
        创建新用户
        """
        now = self._now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    username, email, hashed_password,
                    preferred_role, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username, email, hashed_password, preferred_role, now, now),
            )

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """
        根据用户名查询用户
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """
        根据邮箱查询用户
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
        return dict(row) if row else None

    def update_last_login(self, username: str) -> None:
        """
        更新用户最后登录时间
        """
        now = self._now()
        with self.connection() as conn:
            conn.execute(
                "UPDATE users SET last_login_at = ?, updated_at = ? WHERE username = ?",
                (now, now, username),
            )

    def update_preferred_role(self, username: str, role_id: str) -> None:
        """
        更新用户偏好的角色
        """
        now = self._now()
        with self.connection() as conn:
            conn.execute(
                "UPDATE users SET preferred_role = ?, updated_at = ? WHERE username = ?",
                (role_id, now, username),
            )

    # =========================
    # 消息（对话历史）相关操作
    # =========================

    def save_message(
        self,
        username: str,
        role_id: str,
        message_role: str,
        content: str,
    ) -> None:
        """
        保存一条对话消息
        """
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    username, role_id, message_role, content, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, role_id, message_role, content, self._now()),
            )

    def delete_role_messages(self, username: str, role_id: str) -> None:
        """
        删除某个用户在特定角色下的全部对话历史
        """
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM messages WHERE username = ? AND role_id = ?",
                (username, role_id),
            )

    def get_role_history(
        self,
        username: str,
        role_id: str,
        limit: int = 40,
    ) -> List[Dict]:
        """
        获取某个角色下的对话历史（按时间正序）

        注意：数据库中按 id DESC 取，返回时反转以保证时间顺序
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT message_role, content, created_at
                FROM messages
                WHERE username = ? AND role_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (username, role_id, limit),
            ).fetchall()

        return [
            {
                "role": row["message_role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]

    def get_recent_messages(
        self,
        username: str,
        role_id: str,
        limit: int = 10,
    ) -> List[Dict]:
        """
        获取最近几条对话（用于 RAG 上下文）
        """
        history = self.get_role_history(username, role_id, limit=limit)
        return [{"role": item["role"], "content": item["content"]} for item in history]

    def get_all_role_histories(
        self,
        username: str,
        role_ids: List[str],
        per_role_limit: int = 40,
    ) -> Dict[str, List[Dict]]:
        """
        获取用户在所有角色下的对话历史
        """
        return {
            role_id: self.get_role_history(username, role_id, limit=per_role_limit)
            for role_id in role_ids
        }

    def get_conversation_summaries(
        self,
        username: str,
        role_ids: List[str],
    ) -> Dict[str, Dict]:
        """
        获取每个角色会话的摘要信息
        """
        summaries: Dict[str, Dict] = {}

        with self.connection() as conn:
            for role_id in role_ids:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS message_count, MAX(created_at) AS updated_at
                    FROM messages
                    WHERE username = ? AND role_id = ?
                    """,
                    (username, role_id),
                ).fetchone()

                last_row = conn.execute(
                    """
                    SELECT content, message_role, created_at
                    FROM messages
                    WHERE username = ? AND role_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (username, role_id),
                ).fetchone()

                summaries[role_id] = {
                    "message_count": int(row["message_count"] or 0),
                    "updated_at": row["updated_at"],
                    "last_message": last_row["content"] if last_row else "",
                    "last_role": last_row["message_role"] if last_row else "",
                }

        return summaries

    # =========================
    # 管理员相关操作
    # =========================

    def get_admin_summary(self) -> Dict:
        """
        获取系统级统计数据
        """
        with self.connection() as conn:
            user_count_row = conn.execute(
                "SELECT COUNT(*) AS total FROM users"
            ).fetchone()

            query_count_row = conn.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE message_role = ?",
                ("user",),
            ).fetchone()

            message_count_row = conn.execute(
                "SELECT COUNT(*) AS total FROM messages"
            ).fetchone()

            last_user_row = conn.execute(
                "SELECT MAX(created_at) AS last_created_at FROM users"
            ).fetchone()

            last_query_row = conn.execute(
                "SELECT MAX(created_at) AS last_created_at "
                "FROM messages WHERE message_role = ?",
                ("user",),
            ).fetchone()

        return {
            "user_count": int(user_count_row["total"] or 0),
            "query_count": int(query_count_row["total"] or 0),
            "message_count": int(message_count_row["total"] or 0),
            "latest_user_created_at": last_user_row["last_created_at"],
            "latest_query_created_at": last_query_row["last_created_at"],
        }

    def list_users(self, keyword: str = "", limit: int = 100) -> List[Dict]:
        """
        查询用户列表（支持关键词搜索）
        """
        search = f"%{keyword.strip()}%" if keyword.strip() else "%"

        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    users.username,
                    users.email,
                    users.preferred_role,
                    users.created_at,
                    users.updated_at,
                    users.last_login_at,
                    COALESCE(
                        SUM(CASE WHEN messages.message_role = 'user' THEN 1 ELSE 0 END),
                        0
                    ) AS query_count,
                    MAX(messages.created_at) AS last_message_at
                FROM users
                LEFT JOIN messages ON messages.username = users.username
                WHERE users.username LIKE ? OR users.email LIKE ?
                GROUP BY
                    users.username, users.email, users.preferred_role,
                    users.created_at, users.updated_at, users.last_login_at
                ORDER BY users.created_at DESC
                LIMIT ?
                """,
                (search, search, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def list_user_queries(
        self,
        keyword: str = "",
        username: str = "",
        role_id: str = "",
        limit: int = 200,
    ) -> List[Dict]:
        """
        查询用户的历史提问记录
        """
        clauses = ["message_role = ?"]
        params: List[object] = ["user"]

        if keyword.strip():
            clauses.append("content LIKE ?")
            params.append(f"%{keyword.strip()}%")
        if username.strip():
            clauses.append("username = ?")
            params.append(username.strip())
        if role_id.strip():
            clauses.append("role_id = ?")
            params.append(role_id.strip())

        params.append(limit)
        where_sql = " AND ".join(clauses)

        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, username, role_id, content, created_at
                FROM messages
                WHERE {where_sql}
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [dict(row) for row in rows]