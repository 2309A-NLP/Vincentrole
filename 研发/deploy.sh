#!/usr/bin/env bash
# ================================================================
# AURA MODE 多角色 AI 对话系统 — 部署脚本
# ================================================================
# 用法:
#   ./deploy.sh              — 完整部署（检查 → 安装 → 启动）
#   ./deploy.sh install      — 只安装依赖（venv + pip + 模型）
#   ./deploy.sh start        — 启动服务（Docker → 后端 → 前端）
#   ./deploy.sh stop         — 停止所有服务
#   ./deploy.sh status       — 查看运行状态
#   ./deploy.sh restart      — 重启所有服务
#   ./deploy.sh logs         — 查看后端日志
#   ./deploy.sh daemon       — 后台持续运行（nohup）
# ================================================================

set -euo pipefail
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# ── 颜色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ── 配置（可从环境变量覆盖） ──
BACKEND_PORT="${BACKEND_PORT:-8080}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"
MILVUS_URI="${MILVUS_URI:-http://localhost:19530}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
LLM_API_BASE="${LLM_API_BASE:-https://api.deepseek.com}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-deepseek-chat}"
VENV_DIR="${VENV_DIR:-venv}"

# ── 路径 ──
ENV_FILE="${PROJECT_DIR}/.env"
VENV_PATH="${PROJECT_DIR}/${VENV_DIR}"
BACKEND_LOG="${PROJECT_DIR}/backend.log"
FRONTEND_LOG="${PROJECT_DIR}/frontend.log"

# ================================================================
# 前置检查
# ================================================================
check_prerequisites() {
    info "检查前置依赖..."

    # Python
    if ! command -v python3 &>/dev/null; then
        err "未找到 python3，请先安装 Python 3.10+"
        exit 1
    fi
    PY_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
    info "  Python: $(python3 --version) (${GREEN}OK${NC})"

    # Docker
    if ! command -v docker &>/dev/null; then
        warn "未安装 Docker，Milvus/Redis 将无法自动启动"
        warn "请手动确保 Redis 和 Milvus 已运行"
        DOCKER_AVAILABLE=false
    else
        DOCKER_AVAILABLE=true
        ok "Docker: $(docker --version 2>/dev/null)"
    fi

    # Ollama（可选）
    if command -v ollama &>/dev/null; then
        OLLAMA_AVAILABLE=true
        ok "Ollama 已安装"
    else
        OLLAMA_AVAILABLE=false
    fi

    # 检查 macOS 系统代理
    if [[ "${OSTYPE}" == "darwin"* ]]; then
        if scutil --proxy 2>/dev/null | grep -q "HTTPEnable : 1"; then
            warn "macOS 系统代理已开启（可能影响后端 API 调用）"
            warn "  → http_proxy=$(scutil --proxy 2>/dev/null | grep "HTTPProxy" | awk '{print $3}'):$(scutil --proxy 2>/dev/null | grep "HTTPPort" | awk '{print $3}')"
            warn "  如后端调用异常，建议关闭系统代理或设置 no_proxy=localhost"
        fi
    fi
}

# ================================================================
# 环境变量配置
# ================================================================
setup_env() {
    if [[ -f "${ENV_FILE}" ]]; then
        ok ".env 文件已存在，跳过配置"
        info "  如需重新配置，请删除 ${ENV_FILE} 后重新运行"
        return
    fi

    info "创建 .env 配置文件..."
    cat > "${ENV_FILE}" << 'ENVEOF'
# ── AURA MODE 环境配置 ──
# 复制此文件为 .env 并根据实际情况修改

# 后端服务端口
BACKEND_PORT=8080

# 前端 Streamlit 端口
FRONTEND_PORT=8501

# Milvus 向量数据库
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=character_knowledge_v1

# Redis 短期记忆
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# 大模型配置（DeepSeek）
# 如需切换回 Ollama 本地模型，改为：
#   LLM_API_BASE=http://localhost:11434/v1
#   LLM_MODEL_NAME=deepseek-r1:7b
LLM_API_BASE=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-chat
# ⚠️ 填写你的 DeepSeek API Key（从 https://platform.deepseek.com/api_keys 获取）
#     禁止硬编码到代码中，必须通过此环境变量传入
DEEPSEEK_API_KEY=sk-you...here

# JWT 密钥（首次部署务必修改为随机字符串）
SECRET_KEY=change-this-to-a-random-secret-string
# Token 过期时间（分钟，默认 1440 = 24 小时）
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# 管理员令牌（留空则禁用管理后台；设置后用于 /api/admin/* 接口）
ADMIN_TOKEN=

# MinerU 在线 PDF 解析（可选，留空则使用本地引擎兜底）
# MINERU_API_TOKEN=your-mineru-token-here
MINERU_API_TOKEN=

# 日志级别（DEBUG / INFO / WARNING / ERROR）
LOG_LEVEL=INFO
ENVEOF

    info ".env 文件已创建"
    warn "  ⚠️ 请编辑 ${ENV_FILE} 填写 LLM_API_KEY 和 SECRET_KEY"
    warn "  ⚠️ 当前 SECRET_KEY 为占位值，生产环境务必修改"
}

# ================================================================
# 虚拟环境与依赖安装
# ================================================================
install_deps() {
    info "设置 Python 虚拟环境..."

    if [[ -d "${VENV_PATH}" ]]; then
        ok "虚拟环境已存在: ${VENV_PATH}"
    else
        python3 -m venv "${VENV_PATH}"
        ok "虚拟环境已创建"
    fi

    source "${VENV_PATH}/bin/activate"

    info "升级 pip..."
    pip install --upgrade pip -q

    info "安装 Python 依赖..."
    pip install -r requirements.txt -q

    # 检查关键依赖是否安装成功
    python3 -c "import fastapi; import uvicorn; import streamlit; import pymilvus; import redis; import torch" 2>/dev/null \
        && ok "核心依赖安装成功" \
        || { err "关键依赖安装失败"; exit 1; }

    # 可选 OCR 依赖
    info "安装可选 OCR 依赖..."
    pip install pytesseract -q 2>/dev/null && ok "pytesseract 安装成功" || warn "pytesseract 安装失败（图片解析将不可用）"

    deactivate
}

# ================================================================
# 本地模型下载
# ================================================================
download_models() {
    source "${VENV_PATH}/bin/activate"

    info "检查本地模型..."

    # BGE-m3 embedding 模型（约 2.2GB）
    if python3 -c "
from config import settings
import os
path = settings.LOCAL_BGE_M3_PATH
if os.path.isdir(path) and any(f.endswith('.safetensors') or f.endswith('.bin') for f in os.listdir(path)):
    print('found')
" 2>/dev/null | grep -q found; then
        ok "BGE-m3 embedding 模型已存在"
    else
        info "下载 BGE-m3 embedding 模型（首次下载约 2.2GB，请耐心等待）..."
        python3 -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('BAAI/bge-m3', trust_remote_code=True)
AutoModel.from_pretrained('BAAI/bge-m3', trust_remote_code=True)
" 2>&1 | tail -3
        ok "BGE-m3 模型下载完成"
    fi

    # BGE-reranker 模型（约 2.2GB）
    if python3 -c "
from config import settings
import os
path = settings.LOCAL_BGE_RERANK_PATH
if os.path.isdir(path) and any(f.endswith('.safetensors') or f.endswith('.bin') for f in os.listdir(path)):
    print('found')
" 2>/dev/null | grep -q found; then
        ok "BGE-reranker 模型已存在"
    else
        info "下载 BGE-reranker 模型（首次下载约 2.2GB，请耐心等待）..."
        python3 -c "
from sentence_transformers import CrossEncoder
CrossEncoder('BAAI/bge-reranker-v2-m3')
" 2>&1 | tail -3
        ok "BGE-reranker 模型下载完成"
    fi

    deactivate
}

# ================================================================
# Docker 服务管理
# ================================================================
start_docker() {
    if [[ "${DOCKER_AVAILABLE:-false}" == "false" ]]; then
        warn "Docker 不可用，跳过容器启动"
        warn "请确保 Redis 和 Milvus 已在其他方式下运行"
        return
    fi

    # 检查 Milvus 是否已运行
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'milvus-standalone'; then
        ok "Milvus 已运行 (${MILVUS_URI})"
    else
        info "启动 Milvus standalone..."
        docker compose -p milvus up -d 2>/dev/null || {
            warn "本地 docker compose 配置缺失，请手动启动 Milvus"
            warn "  参考: https://milvus.io/docs/install_standalone-docker.md"
        }
    fi

    # 检查 Redis 是否已运行
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qE '(redis|docker-redis)'; then
        ok "Redis 已运行 (${REDIS_HOST}:${REDIS_PORT})"
    else
        info "启动 Redis..."
        docker run -d --name roleplay-redis \
            -p "${REDIS_PORT}":6379 \
            --restart unless-stopped \
            redis:7-alpine redis-server --save "" --appendonly no 2>/dev/null || {
            warn "Redis 启动失败，可能容器名已存在"
            docker start roleplay-redis 2>/dev/null || true
        }
        ok "Redis 已启动"
    fi

    # 等待服务就绪
    info "等待服务就绪..."
    for i in $(seq 1 30); do
        if curl -s "${MILVUS_URI}/health" >/dev/null 2>&1; then
            ok "Milvus 就绪"
            break
        fi
        if [[ $i -eq 30 ]]; then
            warn "Milvus 未及时就绪，请检查 docker logs"
        fi
        sleep 2
    done

    # 验证 Redis
    if command -v redis-cli &>/dev/null; then
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping >/dev/null 2>&1 \
            && ok "Redis 可连接" \
            || warn "Redis 连接失败"
    fi
}

stop_docker() {
    info "停止 Docker 容器..."
    docker stop roleplay-redis 2>/dev/null && ok "Redis 已停止" || true
    # Milvus 由外部 docker-compose 管理，不自动停止
    info "  Milvus 需在对应目录下执行 docker compose down"
}

# ================================================================
# 数据目录
# ================================================================
setup_data_dirs() {
    mkdir -p "${PROJECT_DIR}/data"
    mkdir -p "${PROJECT_DIR}/knowledge_base"
    ok "数据目录已就绪"
}

# ================================================================
# 应用启动
# ================================================================
start_apps() {
    source "${VENV_PATH}/bin/activate"

    # 加载 .env
    if [[ -f "${ENV_FILE}" ]]; then
        set -a
        source "${ENV_FILE}"
        set +a
    fi

    # 导出环境变量供 config.py 读取
    export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
    export ROLEPLAY_SECRET_KEY="${SECRET_KEY:-change-this-to-a-random-secret-string}"
    export ROLEPLAY_ACCESS_TOKEN_EXPIRE_MINUTES="${ACCESS_TOKEN_EXPIRE_MINUTES:-1440}"
    export ROLEPLAY_ADMIN_TOKEN="${ADMIN_TOKEN:-}"
    export ROLEPLAY_LOG_LEVEL="${LOG_LEVEL:-INFO}"

    # 检查 DeepSeek API Key
    if [[ -z "${DEEPSEEK_API_KEY}" ]]; then
        err "DEEPSEEK_API_KEY 未设置"
        err "  请执行以下命令之一："
        err "    1) 编辑 ${ENV_FILE}，填写 DEEPSEEK_API_KEY=sk-xxx"
        err "    2) 或在外部导出环境变量: export DEEPSEEK_API_KEY=sk-xxx"
        exit 1
    fi

    # 构造连接字符串（如果 .env 中的值被 source 读了就不需要额外处理）
    info "启动配置:"
    info "  后端端口: ${BACKEND_PORT}"
    info "  前端端口: ${FRONTEND_PORT}"
    info "  Milvus: ${MILVUS_URI}"
    info "  Redis: ${REDIS_HOST}:${REDIS_PORT}"
    info "  LLM: ${LLM_API_BASE} / ${LLM_MODEL_NAME}"

    # 停止旧进程
    stop_apps

    # 启动后端（FastAPI）
    info "启动后端服务 (uvicorn)..."
    nohup uvicorn main:app \
        --host 0.0.0.0 \
        --port "${BACKEND_PORT}" \
        --log-level "${LOG_LEVEL,,}" \
        --reload \
        > "${BACKEND_LOG}" 2>&1 &
    BACKEND_PID=$!
    echo "${BACKEND_PID}" > "${PROJECT_DIR}/.backend.pid"
    ok "后端已启动 (PID: ${BACKEND_PID})"

    # 等待后端就绪
    info "等待后端就绪..."
    for i in $(seq 1 30); do
        if curl -s "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
            ok "后端就绪 (http://localhost:${BACKEND_PORT})"
            break
        fi
        if [[ $i -eq 30 ]]; then
            warn "后端启动超时，请检查 ${BACKEND_LOG}"
        fi
        sleep 2
    done

    # 启动前端（Streamlit）
    info "启动前端服务 (Streamlit)..."
    nohup streamlit run app.py \
        --server.port "${FRONTEND_PORT}" \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false \
        > "${FRONTEND_LOG}" 2>&1 &
    FRONTEND_PID=$!
    echo "${FRONTEND_PID}" > "${PROJECT_DIR}/.frontend.pid"
    ok "前端已启动 (PID: ${FRONTEND_PID})"

    # Ollama 检查
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        ok "Ollama 运行正常"
    else
        warn "Ollama 未运行，如需本地 LLM 请先启动: ollama serve"
    fi

    deactivate

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  AURA MODE 部署完成${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "  后端 API:  ${CYAN}http://localhost:${BACKEND_PORT}${NC}"
    echo -e "  前端界面:  ${CYAN}http://localhost:${FRONTEND_PORT}${NC}"
    echo -e "  健康检查:  ${CYAN}http://localhost:${BACKEND_PORT}/health${NC}"
    echo -e "  Swagger:   ${CYAN}http://localhost:${BACKEND_PORT}/docs${NC}"
    echo ""
    echo -e "  查看日志:  ${YELLOW}./deploy.sh logs${NC}"
    echo -e "  停止服务:  ${YELLOW}./deploy.sh stop${NC}"
    echo -e "  重启服务:  ${YELLOW}./deploy.sh restart${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# ================================================================
# 应用停止
# ================================================================
stop_apps() {
    if [[ -f "${PROJECT_DIR}/.backend.pid" ]]; then
        kill "$(cat "${PROJECT_DIR}/.backend.pid")" 2>/dev/null || true
        rm -f "${PROJECT_DIR}/.backend.pid"
        ok "后端已停止"
    fi

    if [[ -f "${PROJECT_DIR}/.frontend.pid" ]]; then
        kill "$(cat "${PROJECT_DIR}/.frontend.pid")" 2>/dev/null || true
        rm -f "${PROJECT_DIR}/.frontend.pid"
        ok "前端已停止"
    fi

    # 清理残留进程
    pkill -f "uvicorn main:app" 2>/dev/null || true
    pkill -f "streamlit run app.py" 2>/dev/null || true
}

# ================================================================
# 状态检查
# ================================================================
check_status() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo -e "${CYAN}  AURA MODE 服务状态${NC}"
    echo -e "${CYAN}═══════════════════════════════════════${NC}"

    # 后端
    if curl -sf "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
        HEALTH=$(curl -sf "http://localhost:${BACKEND_PORT}/health" 2>/dev/null)
        echo -e "  后端 API:     ${GREEN}运行中${NC}  (http://localhost:${BACKEND_PORT})"
        echo -e "  健康状态:     ${HEALTH}"
    else
        echo -e "  后端 API:     ${RED}未运行${NC}"
    fi

    # 前端
    if curl -sf "http://localhost:${FRONTEND_PORT}" >/dev/null 2>&1; then
        echo -e "  前端界面:     ${GREEN}运行中${NC}  (http://localhost:${FRONTEND_PORT})"
    else
        echo -e "  前端界面:     ${RED}未运行${NC}"
    fi

    # Milvus
    if curl -sf "${MILVUS_URI}/health" >/dev/null 2>&1; then
        echo -e "  Milvus:       ${GREEN}运行中${NC}  (${MILVUS_URI})"
    else
        echo -e "  Milvus:       ${RED}未运行${NC}"
    fi

    # Redis
    if command -v redis-cli &>/dev/null; then
        if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping 2>/dev/null | grep -q PONG; then
            echo -e "  Redis:        ${GREEN}运行中${NC}  (${REDIS_HOST}:${REDIS_PORT})"
        else
            echo -e "  Redis:        ${RED}未运行${NC}"
        fi
    fi

    # Ollama
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo -e "  Ollama:       ${GREEN}运行中${NC}"
    else
        echo -e "  Ollama:       ${YELLOW}未运行（可选）${NC}"
    fi

    # PID 文件检查
    echo ""
    if [[ -f "${PROJECT_DIR}/.backend.pid" ]]; then
        BACKEND_PID=$(cat "${PROJECT_DIR}/.backend.pid")
        if kill -0 "${BACKEND_PID}" 2>/dev/null; then
            echo -e "  后端 PID:     ${BACKEND_PID}"
        else
            echo -e "  后端 PID:     ${RED}${BACKEND_PID} (失效)${NC}"
        fi
    fi
    if [[ -f "${PROJECT_DIR}/.frontend.pid" ]]; then
        FRONTEND_PID=$(cat "${PROJECT_DIR}/.frontend.pid")
        if kill -0 "${FRONTEND_PID}" 2>/dev/null; then
            echo -e "  前端 PID:     ${FRONTEND_PID}"
        else
            echo -e "  前端 PID:     ${RED}${FRONTEND_PID} (失效)${NC}"
        fi
    fi
    echo ""
}

# ================================================================
# 日志查看
# ================================================================
show_logs() {
    BOTH=false
    if [[ $# -eq 0 ]]; then
        set -- "backend"
    fi

    case "${1}" in
        backend|back)
            echo -e "${CYAN}=== 后端日志 (${BACKEND_LOG}) ===${NC}"
            tail -f "${BACKEND_LOG}" 2>/dev/null || echo "日志文件不存在"
            ;;
        frontend|front)
            echo -e "${CYAN}=== 前端日志 (${FRONTEND_LOG}) ===${NC}"
            tail -f "${FRONTEND_LOG}" 2>/dev/null || echo "日志文件不存在"
            ;;
        both|all)
            echo -e "${CYAN}=== 后端日志 ===${NC}"
            tail -20 "${BACKEND_LOG}" 2>/dev/null || echo "日志文件不存在"
            echo ""
            echo -e "${CYAN}=== 前端日志 ===${NC}"
            tail -20 "${FRONTEND_LOG}" 2>/dev/null || echo "日志文件不存在"
            ;;
        *)
            err "未知参数: ${1}，可用: backend / frontend / both"
            ;;
    esac
}

# ================================================================
# 一键部署（完整流程）
# ================================================================
full_deploy() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║    AURA MODE 一键部署流程           ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
    echo ""

    check_prerequisites
    setup_env
    install_deps
    setup_data_dirs
    start_docker

    echo ""
    echo -e "${YELLOW}是否下载本地模型（首次部署需要，约 4.5GB，耗时 5-15 分钟）？${NC}"
    echo -e "${YELLOW}选择 y 下载模型，选择 n 跳过（之后可单独运行 ./deploy.sh install）${NC}"
    read -r -p "下载模型? [y/N]: " download_choice
    if [[ "${download_choice}" =~ ^[Yy]$ ]]; then
        download_models
    fi

    start_apps
}

# ================================================================
# 入口
# ================================================================
case "${1:-}" in
    install)
        check_prerequisites
        setup_env
        install_deps
        setup_data_dirs
        download_models
        info "依赖安装完成，请执行 ./deploy.sh start 启动服务"
        ;;
    start)
        check_prerequisites
        start_docker
        start_apps
        ;;
    stop)
        stop_apps
        stop_docker
        ok "所有服务已停止"
        ;;
    restart)
        stop_apps
        sleep 2
        start_apps
        ;;
    status)
        check_status
        ;;
    logs)
        shift
        show_logs "$@"
        ;;
    daemon)
        # 后台运行（nohup 自己）
        info "后台部署模式..."
        nohup bash "$0" start > "${PROJECT_DIR}/deploy_daemon.log" 2>&1 &
        DAEMON_PID=$!
        echo "${DAEMON_PID}" > "${PROJECT_DIR}/.daemon.pid"
        ok "部署守护进程已启动 (PID: ${DAEMON_PID})"
        info "日志: ${PROJECT_DIR}/deploy_daemon.log"
        info "查看状态: ./deploy.sh status"
        ;;
    *)
        full_deploy
        ;;
esac
