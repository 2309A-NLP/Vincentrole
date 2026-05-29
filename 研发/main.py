# -*- coding: utf-8 -*-
"""
FastAPI 后端主入口（Backend Service）

功能概述：
- 提供用户认证（注册 / 登录 / JWT）
- 提供文件上传与知识库摄入（PDF / 图片）
- 提供多角色 AI 对话接口（RAG）
- 提供管理后台接口
- 协调 RAG 核心、向量库、持久化存储等模块
"""

import os
import re
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Optional

# ---------- 环境与配置 ----------
# 关闭 tokenizer 并行，避免与 FastAPI 冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import Depends, FastAPI, Header, HTTPException, Query, UploadFile, File
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import uvicorn

from config import settings
from logging_config import setup_logging
from image_parser import parse_image
from pdf_parser import parse_pdf
from persistence import SQLiteStore
from role_catalog import DEFAULT_ROLE_ID, ROLE_CATALOG, backend_character_prompts
from text_chunking import split_text
import logging

# ---------- 日志初始化 ----------
setup_logging()
logger = logging.getLogger(__name__)

# ---------- 全局配置 ----------
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
ADMIN_TOKEN = settings.ADMIN_TOKEN

# 邮箱正则校验
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

# 密码加密上下文（bcrypt）
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 持久化存储（SQLite）
app_store = SQLiteStore(settings.APP_DB_PATH)

# 角色提示词字典
CHARACTER_DB: Dict[str, str] = backend_character_prompts()

# ---------- 系统资源初始化 ----------
logger.info("Initializing system resources.")
try:
    from rag_core import chat_service
    from embeddings import embed_engine
    from storage import milvus_db

    # 触发模型加载，提前发现错误
    _ = embed_engine.model
    _ = embed_engine.reranker
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.exception("System initialization failed: %s", e)

# ============================================================
# 数据模型（Pydantic）
# ============================================================

class UserCreate(BaseModel):
    username: str
    password: str


class UserRegister(UserCreate):
    email: str


class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    preferred_role: str


class ChatRequest(BaseModel):
    user_id: str
    char_id: str
    query: str


class ChatResponse(BaseModel):
    response: str


class RolePreferenceRequest(BaseModel):
    role_id: str

# ============================================================
# 知识库摄入相关函数
# ============================================================

def ingest_text_content(
    content: str,
    source_name: str,
    source_kind: str,
    parser: str,
    extra_meta: str = ""
) -> int:
    """
    将解析后的文本内容进行分块、向量化并写入 Milvus
    """
    chunks = split_text(content)
    if not chunks:
        raise ValueError("解析成功，但没有可入库的有效文本。")

    # 构造元数据
    meta_parts = [f"user_{source_kind}:{source_name}", f"parser={parser}"]
    if extra_meta:
        meta_parts.append(extra_meta)
    meta = ":".join(meta_parts)
    metas = [meta] * len(chunks)

    batch_size = 16
    logger.info(
        "Ingesting text content: source_kind=%s source=%s parser=%s chunks=%s batch_size=%s",
        source_kind, source_name, parser, len(chunks), batch_size
    )

    # 分批向量化并入库
    for start in range(0, len(chunks), batch_size):
        batch_texts = chunks[start:start + batch_size]
        batch_metas = metas[start:start + batch_size]
        vectors = embed_engine.encode(batch_texts)
        milvus_db.insert_data(texts=batch_texts, vectors=vectors, metas=batch_metas)

        logger.debug(
            "Inserted ingest batch: source=%s start=%s batch_size=%s",
            source_name, start, len(batch_texts)
        )

    return len(chunks)


def ingest_pdf_file(file_path: str, source_name: str) -> Dict:
    """
    解析 PDF 文件并写入知识库
    """
    logger.info("Parsing uploaded PDF: source=%s", source_name)
    result = parse_pdf(file_path)

    chunk_count = ingest_text_content(
        result.content,
        source_name=source_name,
        source_kind="pdf",
        parser=result.parser,
        extra_meta=f"page_count={result.pages}",
    )

    return {
        "filename": source_name,
        "parser": result.parser,
        "pages": result.pages,
        "chunks": chunk_count,
        "warnings": result.warnings,
        "message": "PDF 已解析并写入知识库",
    }


def ingest_image_file(file_path: str, source_name: str) -> Dict:
    """
    解析图片文件并写入知识库
    """
    logger.info("Parsing uploaded image: source=%s", source_name)
    result = parse_image(file_path)

    chunk_count = ingest_text_content(
        result.content,
        source_name=source_name,
        source_kind="image",
        parser=result.parser,
    )

    return {
        "filename": source_name,
        "parser": result.parser,
        "chunks": chunk_count,
        "warnings": result.warnings,
        "message": "图片已解析并写入知识库",
    }

# ============================================================
# 认证与安全工具函数
# ============================================================

def _password_to_bytes(password: str) -> bytes:
    """限制密码长度，兼容 bcrypt"""
    return password.encode("utf-8")[:72] if isinstance(password, str) else password[:72]


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(_password_to_bytes(plain_password), hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(_password_to_bytes(password))


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(email.strip()))


def validate_password_rules(password: str) -> Optional[str]:
    if len(password) < 8 or len(password) > 20:
        return "密码长度需为 8 到 20 位"
    if not re.search(r"[A-Z]", password):
        return "密码必须包含至少一个大写字母"
    if not re.search(r"[a-z]", password):
        return "密码必须包含至少一个小写字母"
    if not re.search(r"\d", password):
        return "密码必须包含至少一个数字"
    return None

# ============================================================
# 依赖注入（FastAPI Dependencies）
# ============================================================

def get_current_user(
    authorization: Optional[str] = Header(default=None, alias="Authorization")
) -> Dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少有效的登录凭证")

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="登录凭证无效")
    except JWTError:
        raise HTTPException(status_code=401, detail="登录凭证已失效，请重新登录")

    user = app_store.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=401, detail="当前用户不存在")
    return user


def require_admin(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")
) -> str:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="当前服务尚未配置管理员后台令牌")
    if not x_admin_token or x_admin_token.strip() != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="管理员令牌无效")
    return x_admin_token

# ============================================================
# Bootstrap 数据构造
# ============================================================

def build_bootstrap_payload(user: Dict) -> Dict:
    role_ids = list(ROLE_CATALOG.keys())
    histories = app_store.get_all_role_histories(user["username"], role_ids, per_role_limit=40)
    conversation_summaries = app_store.get_conversation_summaries(user["username"], role_ids)

    return {
        "username": user["username"],
        "email": user["email"],
        "preferred_role": user.get("preferred_role") or DEFAULT_ROLE_ID,
        "chat_sessions": histories,
        "conversation_summaries": conversation_summaries,
    }

# ============================================================
# FastAPI 应用初始化
# ============================================================

app = FastAPI(title="Character Roleplay RAG System")

# ============================================================
# 用户与认证接口
# ============================================================

@app.post("/api/register", response_model=dict)
async def register(user: UserRegister):
    normalized_email = user.email.strip().lower()
    normalized_username = user.username.strip()

    if not normalized_username:
        raise HTTPException(status_code=400, detail="用户名不能为空")
    if not is_valid_email(normalized_email):
        raise HTTPException(status_code=400, detail="邮箱格式不正确")
    if app_store.get_user_by_username(normalized_username):
        raise HTTPException(status_code=400, detail="用户名已存在")
    if app_store.get_user_by_email(normalized_email):
        raise HTTPException(status_code=400, detail="该邮箱已被注册")

    password_error = validate_password_rules(user.password)
    if password_error:
        raise HTTPException(status_code=400, detail=password_error)

    hashed_password = get_password_hash(user.password)
    app_store.create_user(normalized_username, normalized_email, hashed_password, DEFAULT_ROLE_ID)

    logger.info("User registered: username=%s", normalized_username)
    return {"message": "注册成功"}


@app.post("/api/login", response_model=Token)
async def login(user: UserCreate):
    normalized_username = user.username.strip()
    user_data = app_store.get_user_by_username(normalized_username)

    if not user_data or not verify_password(user.password, user_data["hashed_password"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    app_store.update_last_login(normalized_username)
    access_token = create_access_token(data={"sub": normalized_username})

    logger.info("User logged in: username=%s", normalized_username)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": normalized_username,
        "preferred_role": user_data.get("preferred_role") or DEFAULT_ROLE_ID,
    }

# ============================================================
# 前端初始化接口
# ============================================================

@app.get("/api/bootstrap", response_model=dict)
async def bootstrap(current_user: Dict = Depends(get_current_user)):
    return build_bootstrap_payload(current_user)

# ============================================================
# 角色与历史管理接口
# ============================================================

@app.post("/api/preferences/active-role", response_model=dict)
async def update_active_role(
    payload: RolePreferenceRequest,
    current_user: Dict = Depends(get_current_user)
):
    if payload.role_id not in ROLE_CATALOG:
        raise HTTPException(status_code=404, detail="角色不存在")

    app_store.update_preferred_role(current_user["username"], payload.role_id)
    logger.info(
        "Updated active role: username=%s role_id=%s",
        current_user["username"], payload.role_id
    )
    return {"message": "偏好角色已更新"}


@app.delete("/api/history/{role_id}", response_model=dict)
async def clear_role_history(role_id: str, current_user: Dict = Depends(get_current_user)):
    if role_id not in ROLE_CATALOG:
        raise HTTPException(status_code=404, detail="角色不存在")

    app_store.delete_role_messages(current_user["username"], role_id)
    logger.info(
        "Cleared role history: username=%s role_id=%s",
        current_user["username"], role_id
    )
    return {"message": "当前角色会话已清空"}

# ============================================================
# 知识库文件上传接口
# ============================================================

@app.post("/api/knowledge/upload-pdf", response_model=dict)
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: Dict = Depends(get_current_user)
):
    filename = os.path.basename(file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持上传 PDF 文件")

    content = await file.read()
    max_bytes = 30 * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="PDF 文件不能超过 30MB")
    if not content:
        raise HTTPException(status_code=400, detail="PDF 文件为空")

    logger.info(
        "PDF upload received: username=%s filename=%s bytes=%s",
        current_user["username"], filename, len(content)
    )

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        source_name = f"{current_user['username']}/{filename}"
        return ingest_pdf_file(temp_path, source_name)

    except Exception as exc:
        logger.exception(
            "PDF upload parsing failed: username=%s filename=%s",
            current_user["username"], filename
        )
        raise HTTPException(status_code=500, detail=f"PDF 解析失败：{exc}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/knowledge/upload-image", response_model=dict)
async def upload_image(
    file: UploadFile = File(...),
    current_user: Dict = Depends(get_current_user)
):
    filename = os.path.basename(file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    allowed_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    extension = os.path.splitext(filename)[1].lower()
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="只支持上传 PNG/JPG/JPEG/WEBP/BMP/TIFF 图片")

    content = await file.read()
    max_bytes = 15 * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="图片文件不能超过 15MB")
    if not content:
        raise HTTPException(status_code=400, detail="图片文件为空")

    logger.info(
        "Image upload received: username=%s filename=%s bytes=%s",
        current_user["username"], filename, len(content)
    )

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension or ".png") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        source_name = f"{current_user['username']}/{filename}"
        return ingest_image_file(temp_path, source_name)

    except Exception as exc:
        logger.exception(
            "Image upload parsing failed: username=%s filename=%s",
            current_user["username"], filename
        )
        raise HTTPException(status_code=500, detail=f"图片解析失败：{exc}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ============================================================
# 聊天接口（RAG 核心）
# ============================================================

@app.post("/api/chat", response_model=ChatResponse)
async def handle_chat(
    request: ChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    if request.char_id not in CHARACTER_DB:
        raise HTTPException(status_code=404, detail="角色不存在")
    if request.user_id != current_user["username"]:
        raise HTTPException(status_code=403, detail="当前请求用户与登录凭证不匹配")

    char_prompt = CHARACTER_DB[request.char_id]
    history_messages = app_store.get_recent_messages(
        current_user["username"], request.char_id, limit=10
    )

    try:
        logger.info(
            "Chat request started: username=%s role_id=%s query_length=%s history_messages=%s",
            current_user["username"],
            request.char_id,
            len(request.query or ""),
            len(history_messages),
        )

        reply = await chat_service.chat(
            user_id=request.user_id,
            char_id=request.char_id,
            user_input=request.query,
            character_prompt=char_prompt,
            history_messages=history_messages,
        )

        app_store.save_message(current_user["username"], request.char_id, "user", request.query)
        app_store.save_message(current_user["username"], request.char_id, "assistant", reply)
        app_store.update_preferred_role(current_user["username"], request.char_id)

        logger.info(
            "Chat request completed: username=%s role_id=%s response_length=%s",
            current_user["username"],
            request.char_id,
            len(reply or ""),
        )
        return ChatResponse(response=reply)

    except Exception as e:
        logger.exception(
            "Chat request failed: username=%s role_id=%s",
            current_user["username"], request.char_id
        )
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 健康检查与管理员接口
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "models": "loaded",
        "roles": len(ROLE_CATALOG),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
    }


@app.get("/api/admin/summary", response_model=dict)
async def admin_summary(_: str = Depends(require_admin)):
    return app_store.get_admin_summary()


@app.get("/api/admin/users", response_model=dict)
async def admin_users(
    keyword: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=500),
    _: str = Depends(require_admin),
):
    return {
        "items": app_store.list_users(keyword=keyword, limit=limit),
        "limit": limit,
        "keyword": keyword,
    }


@app.get("/api/admin/queries", response_model=dict)
async def admin_queries(
    keyword: str = Query(default=""),
    username: str = Query(default=""),
    role_id: str = Query(default=""),
    limit: int = Query(default=200, ge=1, le=500),
    _: str = Depends(require_admin),
):
    return {
        "items": app_store.list_user_queries(
            keyword=keyword,
            username=username,
            role_id=role_id,
            limit=limit,
        ),
        "limit": limit,
        "keyword": keyword,
        "username": username,
        "role_id": role_id,
    }

# ============================================================
# 应用启动入口
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)