import os
import re
from html import escape

import requests
import streamlit as st

from role_catalog import DEFAULT_ROLE_ID, frontend_role_config


BACKEND_URL = os.getenv("ROLEPLAY_BACKEND_URL", "http://localhost:8080").rstrip("/")
ADMIN_ENABLED = bool(os.getenv("ROLEPLAY_ADMIN_TOKEN", "").strip())

ROLE_CONFIG = frontend_role_config()

STATUS_LABELS = {
    True: ("服务可用", "status-online"),
    False: ("服务暂不可用", "status-offline"),
}

PROJECT_NAME = "AURA MODE"
PROJECT_TITLE = "AURA MODE 多角色 AI 对话工作台"
PROJECT_SUMMARY = (
    "把健康梳理、情绪陪伴、学习规划、职场表达和文案策划放进同一个入口，"
    "让用户按场景切换更合适的 AI 协作角色。"
)


st.set_page_config(
    page_title="AURA MODE | 多角色 AI 对话工作台",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    defaults = {
        "logged_in": False,
        "admin_logged_in": False,
        "admin_token": "",
        "username": "",
        "user_email": "",
        "access_token": "",
        "active_role": DEFAULT_ROLE_ID,
        "chat_sessions": {},
        "conversation_summaries": {},
        "register_saved_username": "",
        "register_saved_email": "",
        "pending_login_prefill": False,
        "pending_register_reset": False,
        "auth_view": "登录",
        "register_success_notice": "",
        "bootstrap_needed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_styles():
    st.markdown(
        """
        <style>
        *,
        *::before,
        *::after {
            box-sizing: border-box;
        }

        :root {
            --bg-main: #f5f7fb;
            --bg-accent: #edf2ff;
            --card: rgba(255, 255, 255, 0.72);
            --card-strong: rgba(255, 255, 255, 0.92);
            --line: rgba(15, 23, 42, 0.08);
            --text-main: #111827;
            --text-soft: #6b7280;
            --text-inverse: #f7fbff;
            --primary: #0071e3;
            --primary-dark: #0058b0;
            --secondary: #5ac8fa;
            --accent: #34c759;
            --input-bg: rgba(248, 250, 252, 0.92);
            --chat-user: rgba(237, 244, 255, 0.94);
            --shadow: 0 24px 56px rgba(15, 23, 42, 0.10);
            --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 10% 15%, rgba(0, 113, 227, 0.12), transparent 24%),
                radial-gradient(circle at 84% 12%, rgba(90, 200, 250, 0.12), transparent 22%),
                radial-gradient(circle at 80% 76%, rgba(52, 199, 89, 0.08), transparent 22%),
                linear-gradient(180deg, #fbfcfe 0%, #f2f5fb 52%, #f7f8fc 100%);
            color: var(--text-main);
            font-family: var(--font-sans);
        }

        [data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.72);
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.65);
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(0, 113, 227, 0.14), transparent 34%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.82) 0%, rgba(243, 246, 252, 0.92) 100%);
            border-right: 1px solid rgba(15, 23, 42, 0.06);
        }

        [data-testid="stSidebar"] * {
            color: var(--text-main);
        }

        [data-testid="stSidebar"] .stCaption {
            color: rgba(107, 114, 128, 0.88);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        [data-testid="column"] {
            min-width: 0;
        }

        html, body, p, li, label, .stMarkdown, .stText, .stCaption {
            color: var(--text-main);
            font-family: var(--font-sans);
        }

        h1, h2, h3 {
            color: var(--text-main);
        }

        .hero-card,
        .glass-card,
        .metric-card,
        .role-card,
        .empty-state {
            border: 1px solid var(--line);
            background: var(--card);
            backdrop-filter: blur(20px);
            box-shadow: var(--shadow);
            border-radius: 28px;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            padding: 1.8rem 1.9rem;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(243, 247, 255, 0.9));
        }

        .hero-card::after {
            content: "";
            position: absolute;
            inset: auto -8% -34% auto;
            width: 260px;
            height: 260px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(0, 113, 227, 0.14), rgba(90, 200, 250, 0.02) 70%);
            pointer-events: none;
        }

        .hero-kicker {
            color: var(--primary);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .hero-title {
            font-size: 2.7rem;
            line-height: 1.08;
            margin: 0.5rem 0 0.8rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }

        .hero-body {
            color: var(--text-soft);
            font-size: 1rem;
            line-height: 1.8;
            margin: 0;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.15rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.56rem 0.9rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(15, 23, 42, 0.06);
            color: #334155;
            font-size: 0.84rem;
            font-weight: 600;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
        }

        .hero-meta-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1.15rem;
        }

        .hero-meta-card {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            min-height: 104px;
            padding: 0.85rem 0.95rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(15, 23, 42, 0.05);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
        }

        .hero-meta-label {
            color: var(--text-soft);
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .hero-meta-value {
            color: var(--primary-dark);
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.45;
            word-break: break-word;
        }

        .hero-detail-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.3rem;
        }

        .hero-detail-card {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            padding: 1rem 1rem 1.05rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.64);
            border: 1px solid rgba(15, 23, 42, 0.05);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.45);
            backdrop-filter: blur(8px);
            min-height: 226px;
        }

        .hero-detail-label {
            color: var(--text-soft);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.45rem;
        }

        .hero-detail-title {
            color: var(--text-main);
            font-size: 1.06rem;
            font-weight: 800;
            line-height: 1.45;
            margin-bottom: 0.45rem;
        }

        .hero-detail-copy {
            color: var(--text-soft);
            font-size: 0.9rem;
            line-height: 1.65;
        }

        .top-stat-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.95rem;
            margin-bottom: 0.95rem;
        }

        .top-stat-card {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            min-height: 186px;
            padding: 1.15rem 1.15rem 1.05rem;
            border-radius: 26px;
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(244, 247, 252, 0.82));
            box-shadow: var(--shadow);
        }

        .top-stat-label {
            color: var(--text-soft);
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }

        .top-stat-value {
            color: var(--text-main);
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.7rem;
        }

        .top-stat-copy {
            color: var(--text-soft);
            font-size: 0.92rem;
            line-height: 1.65;
            margin-top: auto;
        }

        .metric-card {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 0.55rem;
            height: 100%;
            min-height: 300px;
            padding: 1.35rem 1.35rem 1.25rem;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(244, 247, 252, 0.84));
        }

        .metric-label {
            color: var(--text-soft);
            font-size: 0.86rem;
            margin-bottom: 0;
            min-height: 1.35rem;
        }

        .metric-value {
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.08;
            margin-bottom: 0;
            min-height: 7.4rem;
        }

        .metric-note {
            color: var(--text-soft);
            font-size: 0.9rem;
            line-height: 1.6;
            margin-top: auto;
            min-height: 5.4rem;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }

        .section-note {
            color: var(--text-soft);
            margin-bottom: 0.9rem;
        }

        .role-card {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            padding: 1.2rem 1.15rem;
            min-height: 296px;
            height: 100%;
            margin-bottom: 0.8rem;
        }

        .role-card.active {
            background: linear-gradient(180deg, rgba(240, 246, 255, 0.98), rgba(248, 250, 255, 0.94));
            border-color: rgba(0, 113, 227, 0.24);
            box-shadow: 0 22px 52px rgba(0, 113, 227, 0.10);
        }

        .role-emoji {
            font-size: 2rem;
            margin-bottom: 0.7rem;
        }

        .role-name {
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        .role-tagline,
        .role-style {
            color: var(--text-soft);
            font-size: 0.92rem;
            line-height: 1.65;
        }

        .role-tagline {
            margin-bottom: 0.75rem;
        }

        .role-style {
            margin-top: auto;
        }

        .role-card-anchor {
            display: none;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .role-card-anchor) {
            display: flex;
            flex-direction: column;
            height: 100%;
            min-height: 0;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .role-card-anchor) > div[data-testid="stElementContainer"]:has(.role-card) {
            flex: 1 1 auto;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .role-card-anchor) .stButton {
            margin-top: auto;
        }

        .glass-card {
            padding: 1.25rem 1.3rem;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(245, 247, 251, 0.86));
        }

        .quick-prompt {
            padding: 0.85rem 1rem;
            min-height: 92px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(245, 247, 252, 0.92));
            border: 1px solid rgba(15, 23, 42, 0.06);
            color: var(--text-main);
            font-weight: 500;
        }

        .auth-showcase {
            margin-top: 1.1rem;
            padding: 1.3rem 1.35rem;
            border-radius: 28px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(244, 247, 252, 0.88));
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .auth-showcase-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .auth-showcase-kicker {
            color: var(--primary);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .auth-showcase-title {
            font-size: 1.45rem;
            line-height: 1.2;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0 0 0.35rem;
        }

        .auth-showcase-copy {
            color: var(--text-soft);
            font-size: 0.93rem;
            line-height: 1.72;
            margin: 0;
        }

        .auth-showcase-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 4.8rem;
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            background: rgba(0, 113, 227, 0.08);
            border: 1px solid rgba(0, 113, 227, 0.10);
            color: var(--primary-dark);
            font-size: 0.82rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .auth-showcase-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
            margin-bottom: 1rem;
        }

        .auth-showcase-card {
            min-height: 148px;
            padding: 1rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 23, 42, 0.05);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.68);
        }

        .auth-showcase-card-label {
            color: var(--text-soft);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .auth-showcase-card-title {
            color: var(--text-main);
            font-size: 1rem;
            line-height: 1.45;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .auth-showcase-card-copy {
            color: var(--text-soft);
            font-size: 0.88rem;
            line-height: 1.68;
        }

        .auth-flow-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .auth-flow-step {
            padding: 0.9rem 0.95rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 23, 42, 0.05);
        }

        .auth-flow-step-index {
            color: var(--primary);
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .auth-flow-step-copy {
            color: var(--text-soft);
            font-size: 0.86rem;
            line-height: 1.62;
        }

        .auth-side-panel {
            margin-top: 1rem;
            padding: 1.15rem 1.2rem;
            border-radius: 28px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(244, 247, 252, 0.9));
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .auth-side-panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.9rem;
        }

        .auth-side-panel-title {
            color: var(--text-main);
            font-size: 1.08rem;
            font-weight: 700;
            line-height: 1.35;
        }

        .auth-side-panel-copy {
            color: var(--text-soft);
            font-size: 0.9rem;
            line-height: 1.68;
            margin: 0 0 0.9rem;
        }

        .auth-side-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.48rem 0.82rem;
            border-radius: 999px;
            background: rgba(0, 113, 227, 0.08);
            border: 1px solid rgba(0, 113, 227, 0.10);
            color: var(--primary-dark);
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .auth-side-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .auth-side-card {
            min-height: 132px;
            padding: 0.95rem 1rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(15, 23, 42, 0.05);
        }

        .auth-side-card-label {
            color: var(--text-soft);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .auth-side-card-title {
            color: var(--text-main);
            font-size: 0.98rem;
            line-height: 1.45;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .auth-side-card-copy {
            color: var(--text-soft);
            font-size: 0.86rem;
            line-height: 1.62;
        }

        .chat-shell {
            padding: 1.1rem;
            border-radius: 30px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(243, 246, 252, 0.88));
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.56);
        }

        .chat-composer-anchor {
            display: none;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) {
            padding: 1.15rem 1.15rem 1.2rem;
            border-radius: 24px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.90), rgba(244, 247, 252, 0.88));
            box-shadow: var(--shadow);
            overflow: visible;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) > div {
            min-width: 0;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-testid="column"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-testid="column"] > div {
            min-width: 0 !important;
            max-width: 100% !important;
        }

        .chat-composer-note {
            color: var(--text-soft);
            font-size: 0.88rem;
            line-height: 1.65;
            margin-bottom: 0.55rem;
        }

        .chat-composer-header {
            display: block;
            margin-bottom: 0.9rem;
        }

        .chat-composer-kicker {
            color: var(--primary);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.32rem;
        }

        .chat-composer-title {
            color: var(--text-main);
            font-size: 1.02rem;
            line-height: 1.4;
            font-weight: 700;
            margin-bottom: 0.28rem;
        }

        .chat-composer-copy {
            color: var(--text-soft);
            font-size: 0.9rem;
            line-height: 1.65;
            margin: 0;
        }

        .chat-composer-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            padding: 0.48rem 0.82rem;
            border-radius: 999px;
            background: rgba(0, 113, 227, 0.08);
            border: 1px solid rgba(0, 113, 227, 0.10);
            color: var(--primary-dark);
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
            max-width: 100%;
            margin-top: 0.9rem;
        }

        .knowledge-upload-compact {
            width: 100%;
            max-width: 100%;
            margin: 0.15rem 0 0.85rem;
            padding: 0.9rem 0.95rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.78), rgba(242, 247, 255, 0.72));
            border: 1px solid rgba(0, 113, 227, 0.10);
            overflow: hidden;
        }

        .knowledge-upload-title {
            color: var(--text-main);
            font-size: 0.98rem;
            font-weight: 800;
            line-height: 1.4;
            margin-bottom: 0.28rem;
        }

        .knowledge-upload-copy {
            color: var(--text-soft);
            font-size: 0.86rem;
            line-height: 1.62;
            margin: 0;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-testid="stFileUploader"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            margin: 0.45rem 0 0.7rem;
            overflow: visible;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-testid="stFileUploaderDropzone"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            min-height: 104px;
            border-radius: 18px !important;
            overflow: hidden;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-testid="stFileUploaderDropzone"] > div {
            min-width: 0 !important;
            max-width: 100% !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-testid="stButton"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) .stButton,
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) .stForm,
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-testid="stForm"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-testid="stTextArea"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            overflow: visible;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) .stButton > button,
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) .stForm button {
            border-radius: 18px !important;
            width: 100% !important;
            max-width: 100% !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-testid="stForm"] {
            margin-top: 0;
            padding: 0;
            border: none;
            background: transparent;
            box-shadow: none;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-testid="stTextArea"] {
            margin-bottom: 0.9rem;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-baseweb="textarea"] {
            border-radius: 18px !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(241, 245, 251, 0.94)) !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.92),
                0 10px 26px rgba(15, 23, 42, 0.06) !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) [data-baseweb="textarea"]:focus-within {
            border-color: rgba(0, 113, 227, 0.22) !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.96),
                0 0 0 0.2rem rgba(0, 113, 227, 0.10),
                0 14px 28px rgba(0, 113, 227, 0.10) !important;
            transform: translateY(-1px);
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) .stTextArea textarea {
            min-height: 140px !important;
            padding: 1.15rem 1.2rem !important;
            border: none !important;
            border-radius: 18px !important;
            background: transparent !important;
            box-shadow: none !important;
            color: var(--text-main) !important;
            font-size: 1rem !important;
            line-height: 1.75 !important;
            resize: vertical;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) .stTextArea textarea:focus,
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) .stTextArea textarea:focus-visible {
            outline: none !important;
            border: none !important;
            box-shadow: none !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .chat-composer-anchor) .stTextArea textarea::placeholder {
            color: #9aa8c3 !important;
            font-weight: 600;
        }

        .empty-state {
            padding: 1.6rem;
            text-align: center;
            color: var(--text-soft);
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.78rem;
            border-radius: 999px;
            font-size: 0.88rem;
            margin-top: 0.35rem;
        }

        .status-chip::before {
            content: "";
            width: 0.56rem;
            height: 0.56rem;
            border-radius: 999px;
            display: inline-block;
        }

        .status-online {
            color: #0f766e;
            background: rgba(20, 184, 166, 0.12);
            border: 1px solid rgba(20, 184, 166, 0.18);
        }

        .status-online::before {
            background: #14b8a6;
        }

        .status-offline {
            color: #be123c;
            background: rgba(251, 113, 133, 0.12);
            border: 1px solid rgba(251, 113, 133, 0.18);
        }

        .status-offline::before {
            background: #fb7185;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 22px;
        }

        [data-testid="stForm"] {
            margin-top: 0.9rem;
            padding: 1.2rem 1.2rem 1.1rem;
            border-radius: 28px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(245, 247, 252, 0.9));
            box-shadow: 0 20px 48px rgba(15, 23, 42, 0.08);
        }

        div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(99, 115, 150, 0.10);
            border-radius: 20px;
            padding: 0.25rem 0.45rem;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] li {
            color: var(--text-main);
            line-height: 1.8;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(135deg, rgba(232, 242, 255, 0.96), rgba(235, 251, 247, 0.92));
        }

        .stTextInput label,
        .stChatInput label,
        .stTextArea label,
        .stSelectbox label,
        .stMultiSelect label {
            color: var(--text-main) !important;
            font-weight: 700;
        }

        .stTextInput input,
        .stChatInput input,
        .stTextArea textarea {
            background: var(--input-bg) !important;
            color: var(--text-main) !important;
            border: 1px solid rgba(15, 23, 42, 0.08) !important;
            border-radius: 18px !important;
            min-height: 3rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.88);
        }

        .stTextInput input::placeholder,
        .stChatInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #8a97b2 !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 22px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(16px);
        }

        details {
            border-radius: 22px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            background: rgba(255, 255, 255, 0.74);
            padding: 0.15rem 0.2rem;
        }

        summary {
            font-weight: 600;
            color: var(--text-main);
        }

        .stTabs {
            margin-top: 0.85rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            width: 100%;
            display: grid !important;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            padding: 0.45rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.44);
        }

        .stTabs [data-baseweb="tab"] {
            min-height: 3.4rem;
            margin: 0 !important;
            border-radius: 18px;
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-main);
            font-size: 1.18rem;
            font-weight: 800;
            justify-content: center;
            transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(20, 184, 166, 0.12)) !important;
            color: var(--primary-dark) !important;
            border-color: rgba(59, 130, 246, 0.18);
            box-shadow: 0 10px 22px rgba(59, 130, 246, 0.12);
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.74);
            border-color: rgba(148, 163, 184, 0.18);
            transform: translateY(-1px);
        }

        .stTabs [role="tabpanel"] {
            padding-top: 0.9rem;
        }

        .auth-form-note {
            color: var(--text-soft);
            font-size: 0.9rem;
            line-height: 1.65;
            margin: 0 0 0.45rem;
        }

        .auth-shell {
            margin-top: 1rem;
            padding: 1.1rem;
            border-radius: 30px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(244, 247, 252, 0.9));
            box-shadow: var(--shadow);
            backdrop-filter: blur(22px);
        }

        .auth-shell-header {
            padding: 0.45rem 0.35rem 0.95rem;
        }

        .auth-shell-eyebrow {
            color: var(--primary);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .auth-shell-title {
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0.45rem 0 0.4rem;
        }

        .auth-shell-copy {
            color: var(--text-soft);
            font-size: 0.95rem;
            line-height: 1.7;
            margin: 0;
        }

        .auth-summary-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .auth-summary-card {
            padding: 0.9rem 0.95rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(15, 23, 42, 0.05);
        }

        .auth-summary-label {
            color: var(--text-soft);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .auth-summary-value {
            color: var(--text-main);
            font-size: 1rem;
            line-height: 1.5;
            font-weight: 600;
        }

        .auth-segmented-wrap {
            margin: 0.2rem 0 0.85rem;
            padding: 0.45rem;
            border-radius: 24px;
            background: rgba(248, 250, 252, 0.9);
            border: 1px solid rgba(15, 23, 42, 0.05);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.84);
        }

        .auth-segmented-note {
            color: var(--text-soft);
            font-size: 0.84rem;
            line-height: 1.6;
            margin: 0.45rem 0 0.1rem;
        }

        .auth-shell-anchor {
            display: none;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) {
            padding: 1.15rem;
            border-radius: 30px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(244, 247, 252, 0.9));
            box-shadow: var(--shadow);
            backdrop-filter: blur(22px);
            width: 100%;
            max-width: 100%;
            min-width: 0;
            overflow: hidden;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) > div {
            width: 100%;
            max-width: 100%;
            min-width: 0;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stHorizontalBlock"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="column"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stElementContainer"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) .stMarkdown,
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) .stButton,
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stTextInput"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stCheckbox"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stForm"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stAlert"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-baseweb="base-input"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-baseweb="input"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stTextInputRootElement"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-baseweb="base-input"],
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stTextInputRootElement"] {
            overflow: hidden;
            border-radius: 24px;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stForm"] {
            margin-top: 0.8rem;
            overflow: hidden;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stTextInput"] label,
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stCheckbox"] label {
            color: var(--text-main) !important;
            font-size: 0.98rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-baseweb="base-input"] {
            min-height: 4rem;
            border-radius: 24px !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(241, 245, 251, 0.94)) !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.92),
                0 10px 28px rgba(15, 23, 42, 0.06) !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-baseweb="base-input"]:focus-within {
            border-color: rgba(0, 113, 227, 0.22) !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.96),
                0 0 0 0.2rem rgba(0, 113, 227, 0.10),
                0 14px 32px rgba(0, 113, 227, 0.10) !important;
            transform: translateY(-1px);
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-baseweb="input"] {
            min-height: 4rem;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-baseweb="input"] > div {
            background: transparent !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) .stTextInput input {
            min-height: 4rem !important;
            padding: 0 1.15rem !important;
            border: none !important;
            border-radius: 24px !important;
            background: transparent !important;
            box-shadow: none !important;
            color: var(--text-main) !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) .stTextInput input:focus,
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) .stTextInput input:focus-visible {
            outline: none !important;
            border: none !important;
            box-shadow: none !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) .stTextInput input::placeholder {
            color: #9aa8c3 !important;
            font-weight: 600;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-baseweb="input"] button {
            min-width: 3.5rem;
            margin: 0.34rem 0.4rem 0.34rem 0;
            border: none !important;
            border-radius: 18px !important;
            background: linear-gradient(180deg, rgba(232, 240, 255, 0.96), rgba(219, 232, 255, 0.92)) !important;
            color: var(--primary-dark) !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.92),
                0 6px 16px rgba(59, 130, 246, 0.10) !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-baseweb="input"] button:hover {
            background: linear-gradient(180deg, rgba(222, 235, 255, 1), rgba(208, 226, 255, 0.96)) !important;
            color: var(--primary-dark) !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-baseweb="input"] button svg {
            fill: currentColor !important;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stCheckbox"] {
            margin: 0.2rem 0 0.35rem;
            padding: 0.15rem 0.1rem;
        }

        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) [data-testid="stCheckbox"] input[type="checkbox"] {
            accent-color: #0071e3;
        }

        .saved-login-note,
        .inline-note {
            margin: 0 0 0.8rem;
            padding: 0.9rem 1rem;
            border-radius: 20px;
            border: 1px solid rgba(0, 113, 227, 0.10);
            background: rgba(0, 113, 227, 0.06);
            color: var(--text-soft);
            font-size: 0.9rem;
            line-height: 1.7;
            overflow-wrap: anywhere;
        }

        .password-strength {
            margin: 0.65rem 0 0.8rem;
            padding: 0.85rem 0.95rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(255, 255, 255, 0.72);
            display: block;
            width: 100%;
            max-width: 100%;
            min-width: 0;
            box-sizing: border-box;
            overflow: hidden;
        }

        .password-strength-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.6rem;
            width: 100%;
            max-width: 100%;
            min-width: 0;
            flex-wrap: wrap;
        }

        .password-strength-label {
            color: var(--text-soft);
            font-size: 0.84rem;
            font-weight: 700;
        }

        .password-strength-value {
            font-size: 0.92rem;
            font-weight: 800;
        }

        .strength-empty .password-strength-value {
            color: #94a3b8;
        }

        .strength-weak .password-strength-value {
            color: #dc2626;
        }

        .strength-medium .password-strength-value {
            color: #d97706;
        }

        .strength-strong .password-strength-value {
            color: #2563eb;
        }

        .strength-very-strong .password-strength-value {
            color: #0f766e;
        }

        .password-strength-bar {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.4rem;
            margin-bottom: 0.7rem;
            width: 100%;
            max-width: 100%;
            min-width: 0;
        }

        .password-strength-segment {
            height: 0.45rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.18);
        }

        .password-strength-segment.active-1,
        .password-strength-segment.active-2,
        .password-strength-segment.active-3,
        .password-strength-segment.active-4 {
            background: #dc2626;
        }

        .strength-medium .password-strength-segment.active-1,
        .strength-medium .password-strength-segment.active-2,
        .strength-medium .password-strength-segment.active-3 {
            background: #f59e0b;
        }

        .strength-strong .password-strength-segment.active-1,
        .strength-strong .password-strength-segment.active-2,
        .strength-strong .password-strength-segment.active-3 {
            background: #3b82f6;
        }

        .strength-very-strong .password-strength-segment.active-1,
        .strength-very-strong .password-strength-segment.active-2,
        .strength-very-strong .password-strength-segment.active-3,
        .strength-very-strong .password-strength-segment.active-4 {
            background: #14b8a6;
        }

        .password-checklist {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem 0.75rem;
            width: 100%;
            max-width: 100%;
            min-width: 0;
        }

        .password-check {
            color: var(--text-soft);
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .password-check.pass {
            color: #0f766e;
        }

        .terms-note {
            color: var(--text-soft);
            font-size: 0.84rem;
            line-height: 1.65;
            margin-top: 0.2rem;
        }

        .email-hint {
            margin: 0.45rem 0 0.55rem;
            padding: 0.75rem 0.85rem;
            border-radius: 16px;
            font-size: 0.84rem;
            line-height: 1.6;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(255, 255, 255, 0.68);
            color: var(--text-soft);
            display: block;
            width: 100%;
            max-width: 100%;
            min-width: 0;
            box-sizing: border-box;
            overflow: hidden;
            overflow-wrap: anywhere;
        }

        .email-hint.info {
            color: #2563eb;
            border-color: rgba(59, 130, 246, 0.16);
            background: rgba(59, 130, 246, 0.08);
        }

        .email-hint.success {
            color: #0f766e;
            border-color: rgba(20, 184, 166, 0.18);
            background: rgba(20, 184, 166, 0.08);
        }

        .email-hint.warn {
            color: #b45309;
            border-color: rgba(245, 158, 11, 0.20);
            background: rgba(245, 158, 11, 0.08);
        }

        .stButton > button,
        .stForm button {
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            font-weight: 600;
            min-height: 3rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            letter-spacing: -0.01em;
        }

        .stButton > button[kind="primary"],
        .stForm button[kind="primary"] {
            background: linear-gradient(180deg, #1593ff, #0071e3);
            color: var(--text-inverse);
            box-shadow: 0 14px 28px rgba(0, 113, 227, 0.24);
        }

        .stButton > button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.9);
            color: var(--text-main);
        }

        .stButton > button:hover,
        .stForm button:hover {
            transform: translateY(-1px);
            border-color: rgba(0, 113, 227, 0.22);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
        }

        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.84) !important;
            color: var(--text-main) !important;
            border: 1px solid rgba(15, 23, 42, 0.06) !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }

        [data-testid="stSidebar"] .stButton > button * {
            color: var(--text-main) !important;
            fill: var(--text-main) !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.96) !important;
            color: var(--text-main) !important;
            border-color: rgba(0, 113, 227, 0.14) !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover *,
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover * {
            color: var(--text-main) !important;
            fill: var(--text-main) !important;
        }

        [data-testid="stSidebar"] .stButton > button:focus,
        [data-testid="stSidebar"] .stButton > button:focus-visible,
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:focus,
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:focus-visible {
            color: var(--text-main) !important;
            border-color: rgba(0, 113, 227, 0.18) !important;
            box-shadow: 0 0 0 0.2rem rgba(0, 113, 227, 0.12);
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {
            color: var(--text-main);
        }

        .sidebar-detail {
            margin: 0.35rem 0 0.9rem;
            padding: 0.9rem 0.95rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(191, 219, 254, 0.14);
        }

        .sidebar-hero-card,
        .sidebar-panel,
        .sidebar-tip-card {
            border-radius: 26px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.78), rgba(244, 247, 252, 0.84));
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(16px);
        }

        .sidebar-hero-card {
            position: relative;
            overflow: hidden;
            padding: 1.1rem 1.05rem 1rem;
            margin-bottom: 1rem;
        }

        .sidebar-hero-card::after {
            content: "";
            position: absolute;
            right: -42px;
            top: -32px;
            width: 130px;
            height: 130px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(0, 113, 227, 0.18), rgba(90, 200, 250, 0.01) 72%);
            pointer-events: none;
        }

        .sidebar-kicker {
            color: var(--primary);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .sidebar-title {
            color: var(--text-main);
            font-size: 1.45rem;
            line-height: 1.2;
            font-weight: 700;
            margin: 0.55rem 0 0.5rem;
        }

        .sidebar-copy {
            color: var(--text-soft);
            font-size: 0.95rem;
            line-height: 1.7;
            margin: 0;
        }

        .sidebar-panel {
            padding: 0.95rem;
            margin-bottom: 0.95rem;
        }

        .sidebar-panel-title {
            color: var(--text-main);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }

        .sidebar-info-grid,
        .sidebar-stat-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .sidebar-info-card,
        .sidebar-stat-card {
            padding: 0.85rem 0.85rem 0.9rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 23, 42, 0.05);
        }

        .sidebar-info-card.full,
        .sidebar-stat-card.full {
            grid-column: 1 / -1;
        }

        .sidebar-card-label {
            color: #6b7280;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .sidebar-card-value {
            color: var(--text-main);
            font-size: 1.2rem;
            line-height: 1.45;
            font-weight: 700;
            word-break: break-word;
        }

        .sidebar-card-copy {
            color: var(--text-soft);
            font-size: 0.88rem;
            line-height: 1.65;
            margin-top: 0.35rem;
        }

        .sidebar-tip-card {
            padding: 0.95rem;
            margin: 0.95rem 0 1rem;
        }

        .sidebar-tip-emoji {
            font-size: 1.1rem;
            margin-right: 0.28rem;
        }

        [data-testid="stSidebar"] a {
            color: var(--primary) !important;
        }

        .sidebar-detail-label {
            color: rgba(219, 234, 254, 0.78);
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 0.38rem;
        }

        .sidebar-detail-value {
            color: #f8fbff;
            font-size: 1.15rem;
            line-height: 1.6;
            font-weight: 700;
            white-space: normal;
            word-break: break-word;
        }

        .footer-note {
            color: var(--text-soft);
            font-size: 0.84rem;
            text-align: center;
            margin-top: 1.4rem;
        }

        /* Override Streamlit dark default controls with the app's light glass style. */
        .stTextArea [data-baseweb="textarea"],
        .stTextArea [data-baseweb="base-input"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(241, 245, 251, 0.96)) !important;
            border: 1px solid rgba(148, 163, 184, 0.20) !important;
            border-radius: 24px !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.95),
                0 12px 28px rgba(15, 23, 42, 0.06) !important;
        }

        .stTextArea [data-baseweb="textarea"]:focus-within,
        .stTextArea [data-baseweb="base-input"]:focus-within {
            border-color: rgba(0, 113, 227, 0.28) !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.96),
                0 0 0 0.2rem rgba(0, 113, 227, 0.10),
                0 16px 34px rgba(0, 113, 227, 0.10) !important;
        }

        .stTextArea textarea,
        .stTextArea textarea:focus,
        .stTextArea textarea:focus-visible {
            background: transparent !important;
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
            caret-color: var(--primary) !important;
            outline: none !important;
            box-shadow: none !important;
        }

        .stTextArea textarea::placeholder {
            color: #8a97b2 !important;
            -webkit-text-fill-color: #8a97b2 !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(241, 245, 251, 0.94)) !important;
            border: 1px dashed rgba(0, 113, 227, 0.28) !important;
            border-radius: 18px !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.9),
                0 12px 28px rgba(15, 23, 42, 0.06) !important;
            min-height: 86px;
            padding: 0.75rem 0.9rem !important;
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            overflow: hidden !important;
        }

        [data-testid="stFileUploaderDropzone"] > div {
            display: flex !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 0.9rem !important;
            min-width: 0 !important;
            max-width: 100% !important;
            overflow: hidden !important;
        }

        [data-testid="stFileUploaderDropzone"] > div > div {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            max-width: 100% !important;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(0, 113, 227, 0.42) !important;
            background: linear-gradient(180deg, rgba(247, 251, 255, 0.98), rgba(235, 244, 255, 0.96)) !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.95),
                0 16px 34px rgba(0, 113, 227, 0.10) !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
            min-width: 0 !important;
            max-width: 100% !important;
        }

        [data-testid="stFileUploaderDropzone"] svg {
            color: var(--primary) !important;
            stroke: var(--primary) !important;
            fill: none !important;
        }

        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] + div,
        [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] + span,
        [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] + small {
            display: none !important;
            visibility: hidden !important;
        }

        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
            background: linear-gradient(180deg, rgba(232, 240, 255, 0.98), rgba(218, 232, 255, 0.94)) !important;
            border: 1px solid rgba(0, 113, 227, 0.16) !important;
            border-radius: 16px !important;
            color: var(--primary-dark) !important;
            -webkit-text-fill-color: var(--primary-dark) !important;
            font-weight: 800 !important;
            box-shadow: 0 8px 18px rgba(0, 113, 227, 0.10) !important;
            white-space: nowrap !important;
            flex: 0 0 auto !important;
            width: 7.5rem !important;
            max-width: 7.5rem !important;
            min-height: 2.75rem !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]:hover {
            background: linear-gradient(180deg, rgba(221, 235, 255, 1), rgba(205, 224, 255, 0.98)) !important;
            border-color: rgba(0, 113, 227, 0.28) !important;
        }

        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
            font-size: 0 !important;
        }

        [data-testid="stFileUploaderDropzone"] button::before,
        [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]::before {
            content: "选择文件";
            font-size: 0.92rem;
            line-height: 1;
        }

        [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {
            font-size: 0 !important;
            line-height: 0 !important;
            display: flex !important;
            align-items: center !important;
        }

        [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"]::before {
            content: "拖拽 PDF 或图片到这里";
            display: block;
            color: var(--text-main);
            -webkit-text-fill-color: var(--text-main);
            font-size: 0.96rem;
            line-height: 1.35;
            font-weight: 800;
            white-space: normal;
        }

        [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"]::after {
            content: none;
            display: none;
        }

        [data-testid="stFileUploaderDropzone"] section,
        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] p {
            overflow-wrap: anywhere;
        }

        .stButton [data-testid="stBaseButton-primary"],
        .stForm [data-testid="stBaseButton-primary"],
        .stButton > button[kind="primary"],
        .stForm button[kind="primary"] {
            background: linear-gradient(180deg, #1593ff, #0071e3) !important;
            color: white !important;
            -webkit-text-fill-color: white !important;
            border-color: rgba(0, 113, 227, 0.32) !important;
            box-shadow: 0 12px 24px rgba(0, 113, 227, 0.22) !important;
        }

        @media (max-width: 900px) {
            .hero-title {
                font-size: 2.15rem;
            }

            .hero-meta-grid,
            .hero-detail-grid,
            .auth-summary-grid,
            .auth-showcase-grid,
            .auth-flow-strip,
            .auth-side-grid,
            .password-checklist,
            .sidebar-info-grid,
            .sidebar-stat-grid {
                grid-template-columns: 1fr;
            }

            .auth-shell,
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .auth-shell-anchor) {
                padding: 0.95rem;
                border-radius: 24px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_error_message(response):
    try:
        data = response.json()
    except ValueError:
        return f"请求失败：{response.status_code}"
    return data.get("detail") or data.get("message") or f"请求失败：{response.status_code}"


def is_valid_email(email):
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email.strip()))


def validate_password_rule(password):
    if len(password) < 8 or len(password) > 20:
        return "密码长度需控制在 8 到 20 位之间。"
    if not re.search(r"[A-Z]", password):
        return "密码里至少要有 1 个大写字母。"
    if not re.search(r"[a-z]", password):
        return "密码里至少要有 1 个小写字母。"
    if not re.search(r"\d", password):
        return "密码里至少要有 1 个数字。"
    return None


def get_password_strength(password):
    if not password:
        return {
            "score": 0,
            "label": "尚未设置",
            "class_name": "strength-empty",
            "checks": [],
        }

    checks = [
        ("长度 8-20 位", 8 <= len(password) <= 20),
        ("包含大写字母", bool(re.search(r"[A-Z]", password))),
        ("包含小写字母", bool(re.search(r"[a-z]", password))),
        ("包含数字", bool(re.search(r"\d", password))),
    ]
    score = sum(1 for _, passed in checks if passed)

    if score <= 1:
        label = "弱"
        class_name = "strength-weak"
    elif score == 2:
        label = "中"
        class_name = "strength-medium"
    elif score == 3:
        label = "强"
        class_name = "strength-strong"
    else:
        label = "很强"
        class_name = "strength-very-strong"

    return {
        "score": score,
        "label": label,
        "class_name": class_name,
        "checks": checks,
    }


def get_email_hint(email):
    value = email.strip()
    if not value:
        return "info", "建议填写常用邮箱，后续登录识别和账号找回会更顺畅。"
    if " " in email:
        return "warn", "邮箱里不需要空格，复制粘贴后可以再看一眼。"
    if "@" not in value:
        return "warn", "常见格式是 `name@example.com`，你这里还少一个 `@`。"

    local_part, _, domain_part = value.partition("@")
    if not local_part:
        return "warn", "`@` 前面还需要补上邮箱用户名。"
    if not domain_part:
        return "warn", "`@` 后面还需要完整域名，比如 `icloud.com` 或 `gmail.com`。"

    domain_suggestions = {
        "qq": "qq.com",
        "gmail": "gmail.com",
        "gamil.com": "gmail.com",
        "gmial.com": "gmail.com",
        "163": "163.com",
        "126": "126.com",
        "outlook": "outlook.com",
        "hotmail": "hotmail.com",
        "yahoo": "yahoo.com",
        "qq.con": "qq.com",
        "163.con": "163.com",
        "gmail.con": "gmail.com",
    }

    normalized_domain = domain_part.lower()
    if normalized_domain in domain_suggestions:
        suggestion = domain_suggestions[normalized_domain]
        if suggestion != normalized_domain:
            return "warn", f"这个域名看起来像是想输入 `{suggestion}`。"
        return "warn", f"域名还没写完整，常见写法是 `{suggestion}`。"

    if "." not in domain_part:
        return "warn", f"域名部分还没写完整，通常会像 `{domain_part}.com` 这样。"

    if normalized_domain.endswith(".con"):
        return "warn", f"后缀看起来像手滑了，通常应为 `{domain_part[:-4]}.com`。"

    if is_valid_email(value):
        return "success", "邮箱格式正确，可以继续。"

    return "warn", "这个邮箱格式还不太标准，建议再检查一下 `@`、域名和后缀。"


def request_json(method, path, payload=None, timeout=20, auth_required=False, params=None, extra_headers=None):
    url = f"{BACKEND_URL}{path}"
    headers = {}
    if auth_required and st.session_state.get("access_token"):
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    if extra_headers:
        headers.update(extra_headers)
    response = requests.request(method=method, url=url, json=payload, params=params, timeout=timeout, headers=headers)
    return response


def upload_knowledge_file_request(uploaded_file, endpoint, mime_type, timeout=180):
    url = f"{BACKEND_URL}{endpoint}"
    headers = {}
    if st.session_state.get("access_token"):
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            mime_type,
        )
    }
    return requests.post(url, files=files, headers=headers, timeout=timeout)


def build_default_chat_sessions():
    return {
        role_id: [{"role": "assistant", "content": role["intro"]}]
        for role_id, role in ROLE_CONFIG.items()
    }


def format_summary_time(value):
    if not value:
        return "尚未开始"
    if value == "刚刚":
        return value
    value = str(value).replace("T", " ")
    return value[:16]


def safe_text(value):
    return escape(str(value or ""))


def apply_bootstrap_payload(payload):
    st.session_state.username = payload.get("username", st.session_state.username)
    st.session_state.user_email = payload.get("email", "")

    histories = build_default_chat_sessions()
    incoming_histories = payload.get("chat_sessions", {})
    for role_id, role in ROLE_CONFIG.items():
        history = incoming_histories.get(role_id) or []
        normalized_history = [{"role": item["role"], "content": item["content"]} for item in history if item.get("content")]
        histories[role_id] = normalized_history or [{"role": "assistant", "content": role["intro"]}]

    preferred_role = payload.get("preferred_role", DEFAULT_ROLE_ID)
    summaries = payload.get("conversation_summaries", {})
    for summary in summaries.values():
        summary["updated_at"] = format_summary_time(summary.get("updated_at"))

    st.session_state.chat_sessions = histories
    st.session_state.active_role = preferred_role if preferred_role in ROLE_CONFIG else DEFAULT_ROLE_ID
    st.session_state.conversation_summaries = summaries
    st.session_state.bootstrap_needed = False


def bootstrap_user_data():
    response = request_json("GET", "/api/bootstrap", timeout=20, auth_required=True)
    if response.status_code == 200:
        apply_bootstrap_payload(response.json())
        return True

    if response.status_code == 401:
        st.session_state.logged_in = False
        st.session_state.access_token = ""
        st.session_state.username = ""
        st.session_state.user_email = ""
        st.session_state.chat_sessions = {}
        st.session_state.conversation_summaries = {}
        st.session_state.bootstrap_needed = False
    return False


def get_admin_headers():
    token = st.session_state.get("admin_token", "").strip()
    return {"X-Admin-Token": token} if token else {}


def admin_request(path, params=None, timeout=20):
    return request_json("GET", path, timeout=timeout, params=params, extra_headers=get_admin_headers())


def clear_admin_session():
    st.session_state.admin_logged_in = False
    st.session_state.admin_token = ""


@st.cache_data(ttl=20, show_spinner=False)
def get_backend_status(url):
    try:
        response = requests.get(f"{url}/health", timeout=4)
        if response.status_code == 200:
            return True, response.json()
    except requests.RequestException:
        return False, {}
    return False, {}


def get_current_role():
    role_id = st.session_state.active_role
    if role_id not in ROLE_CONFIG:
        role_id = DEFAULT_ROLE_ID
        st.session_state.active_role = role_id
    return role_id, ROLE_CONFIG[role_id]


def ensure_history(role_id):
    chat_sessions = st.session_state.chat_sessions
    if role_id not in chat_sessions or not chat_sessions[role_id]:
        intro = ROLE_CONFIG[role_id]["intro"]
        chat_sessions[role_id] = [{"role": "assistant", "content": intro}]
    return chat_sessions[role_id]


def clear_current_history():
    role_id, role = get_current_role()
    st.session_state.chat_sessions[role_id] = [{"role": "assistant", "content": role["intro"]}]
    if st.session_state.logged_in:
        try:
            request_json("DELETE", f"/api/history/{role_id}", timeout=15, auth_required=True)
        except requests.RequestException:
            pass
    summary = st.session_state.conversation_summaries.get(role_id, {})
    summary.update({"message_count": 0, "last_message": "", "last_role": "", "updated_at": None})
    st.session_state.conversation_summaries[role_id] = summary


def update_local_summary(role_id, last_message, last_role):
    current_history = st.session_state.chat_sessions.get(role_id, [])
    st.session_state.conversation_summaries[role_id] = {
        "message_count": max(len(current_history) - 1, 0),
        "last_message": last_message,
        "last_role": last_role,
        "updated_at": "刚刚",
    }


def submit_message(message):
    content = message.strip()
    if not content:
        return

    role_id, role = get_current_role()
    history = ensure_history(role_id)
    history.append({"role": "user", "content": content})
    update_local_summary(role_id, content, "user")

    try:
        with st.spinner("对方正在接话..."):
            response = request_json(
                "POST",
                "/api/chat",
                payload={
                    "user_id": st.session_state.username,
                    "char_id": role_id,
                    "query": content,
                },
                timeout=90,
                auth_required=True,
            )
        if response.status_code == 200:
            reply = response.json().get("response", "这次暂时没有生成有效回复，请稍后再试。")
            history.append({"role": "assistant", "content": reply})
            update_local_summary(role_id, reply, "assistant")
        else:
            history.append(
                {
                    "role": "assistant",
                    "content": f"消息发送失败，原因是：{get_error_message(response)}",
                }
            )
            update_local_summary(role_id, history[-1]["content"], "assistant")
    except requests.RequestException as exc:
        history.append(
            {
                "role": "assistant",
                "content": f"当前无法连接服务，因此消息未发送成功。具体报错：{exc}",
            }
        )
        update_local_summary(role_id, history[-1]["content"], "assistant")


def show_login_page():
    online, _ = get_backend_status(BACKEND_URL)
    status_text, status_class = STATUS_LABELS[online]

    if st.session_state.pending_register_reset:
        st.session_state.register_email = ""
        st.session_state.register_username = ""
        st.session_state.register_password = ""
        st.session_state.register_confirm_password = ""
        st.session_state.register_show_password = False
        st.session_state.register_agree_terms = False
        st.session_state.pending_register_reset = False

    if st.session_state.pending_login_prefill:
        st.session_state.login_username = st.session_state.register_saved_username
        if "login_password" in st.session_state:
            st.session_state.login_password = ""
        st.session_state.auth_view = "登录"
        st.session_state.pending_login_prefill = False

    auth_view = st.session_state.auth_view
    hero_col, form_col = st.columns([1.2, 1], gap="large")

    with hero_col:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="hero-kicker">{PROJECT_NAME}</div>
                <div class="hero-title">{PROJECT_TITLE}</div>
                <p class="hero-body">
                    {PROJECT_SUMMARY}
                    登录后可以继续上次的角色、偏好与对话节奏，把注意力留给内容本身，而不是重复找入口或重新铺垫。
                </p>
                <div class="badge-row">
                    <span class="badge">五种角色，覆盖不同沟通场景</span>
                    <span class="badge">切换角色即切换回应方式</span>
                    <span class="badge">登录后延续上次会话状态</span>
                    <span class="badge">一个入口完成多场景 AI 协作</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(3, gap="medium")
        metrics = [
            ("上手体验", "打开就会用", "结构更像系统级面板，信息被分层摆好，第一次进来也不容易迷路。"),
            ("角色切换", "像切换模式一样自然", "不是只换名字，而是连回应语气、节奏和建议方式都会一起调整。"),
            ("当前能力", f"{len(ROLE_CONFIG)} 种对话模式", "一个更适合理性梳理，一个更偏陪伴回应，后续也可以继续扩展。"),
        ]
        for col, (label, value, note) in zip(metric_cols, metrics):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-note">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if auth_view == "注册":
            st.markdown(
                """
                <div class="auth-showcase">
                    <div class="auth-showcase-header">
                        <div>
                            <div class="auth-showcase-kicker">New Account</div>
                            <div class="auth-showcase-title">先建立你的 AURA MODE 入口</div>
                            <p class="auth-showcase-copy">
                                注册后可保存账号、角色偏好与对话上下文。以后回来时，
                                系统会更快带你回到熟悉的使用状态。
                            </p>
                        </div>
                        <div class="auth-showcase-badge">注册中</div>
                    </div>
                    <div class="auth-showcase-grid">
                        <div class="auth-showcase-card">
                            <div class="auth-showcase-card-label">账号识别</div>
                            <div class="auth-showcase-card-title">常用邮箱更适合长期使用</div>
                            <div class="auth-showcase-card-copy">用于登录、识别账号和后续找回，建议使用最常用的邮箱。</div>
                        </div>
                        <div class="auth-showcase-card">
                            <div class="auth-showcase-card-label">安全设置</div>
                            <div class="auth-showcase-card-title">满足基础强度即可</div>
                            <div class="auth-showcase-card-copy">系统会实时显示密码强度，保证安全的同时不增加记忆负担。</div>
                        </div>
                    </div>
                    <div class="auth-flow-strip">
                        <div class="auth-flow-step">
                            <div class="auth-flow-step-index">01</div>
                            <div class="auth-flow-step-copy">填写邮箱、用户名与密码，建立账号基础信息。</div>
                        </div>
                        <div class="auth-flow-step">
                            <div class="auth-flow-step-index">02</div>
                            <div class="auth-flow-step-copy">根据实时强度提示调整密码，一次设置到位。</div>
                        </div>
                        <div class="auth-flow-step">
                            <div class="auth-flow-step-index">03</div>
                            <div class="auth-flow-step-copy">注册完成后自动返回登录页，并保留刚填写的账号信息。</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="auth-showcase">
                    <div class="auth-showcase-header">
                        <div>
                            <div class="auth-showcase-kicker">Welcome Back</div>
                            <div class="auth-showcase-title">进入 AURA MODE，继续上次的对话节奏</div>
                            <p class="auth-showcase-copy">
                                这是一个多角色 AI 对话工作台，适合处理健康、情绪、学习、职场和文案等不同场景。
                                登录后会延续上次使用的角色和上下文，不必重新开始。
                            </p>
                        </div>
                        <div class="auth-showcase-badge">登录中</div>
                    </div>
                    <div class="auth-showcase-grid">
                        <div class="auth-showcase-card">
                            <div class="auth-showcase-card-label">多角色协作</div>
                            <div class="auth-showcase-card-title">一个入口切换五种回应模式</div>
                            <div class="auth-showcase-card-copy">健康、灵感、职场、学习、文案场景都能按需切换，保持统一体验。</div>
                        </div>
                        <div class="auth-showcase-card">
                            <div class="auth-showcase-card-label">续接状态</div>
                            <div class="auth-showcase-card-title">常用角色和聊天节奏会被保留</div>
                            <div class="auth-showcase-card-copy">重新进入后可以更快回到熟悉的工作流，减少重复铺垫。</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with form_col:
        auth_header_title = f"登录 {PROJECT_NAME}" if auth_view == "登录" else f"创建 {PROJECT_NAME} 账号"
        auth_header_copy = (
            "进入多角色 AI 对话工作台，继续你的角色偏好与聊天进度。"
            if auth_view == "登录"
            else "注册后即可保存角色偏好、会话记录与常用登录信息。"
        )
        auth_recommendation = "继续当前会话" if auth_view == "登录" else "完成账号创建后开始使用"
        st.markdown(
            f"""
            <div class="auth-shell">
                <div class="auth-shell-header">
                    <div class="auth-shell-eyebrow">Account</div>
                    <div class="auth-shell-title">{auth_header_title}</div>
                    <p class="auth-shell-copy">
                        {auth_header_copy}
                    </p>
                    <div class="auth-summary-grid">
                        <div class="auth-summary-card">
                            <div class="auth-summary-label">当前服务状态</div>
                            <div class="auth-summary-value">{status_text}</div>
                        </div>
                        <div class="auth-summary-card">
                            <div class="auth-summary-label">推荐使用方式</div>
                            <div class="auth-summary-value">{auth_recommendation}</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown('<div class="auth-shell-anchor"></div>', unsafe_allow_html=True)
            toggle_col_login, toggle_col_register = st.columns(2, gap="small")
            with toggle_col_login:
                if st.button(
                    "登录",
                    key="auth_view_login",
                    use_container_width=True,
                    type="primary" if st.session_state.auth_view == "登录" else "secondary",
                ):
                    if st.session_state.auth_view != "登录":
                        st.session_state.auth_view = "登录"
                        st.rerun()
            with toggle_col_register:
                if st.button(
                    "注册",
                    key="auth_view_register",
                    use_container_width=True,
                    type="primary" if st.session_state.auth_view == "注册" else "secondary",
                ):
                    if st.session_state.auth_view != "注册":
                        st.session_state.auth_view = "注册"
                        st.rerun()
            st.markdown(
                f'<div class="auth-segmented-note">{"登录后将直接进入你的角色工作台。" if auth_view == "登录" else "注册后可保存账号信息与角色偏好。"}'
                "</div>",
                unsafe_allow_html=True,
            )

            if auth_view == "登录":
                if st.session_state.register_success_notice:
                    st.success(st.session_state.register_success_notice)
                    st.session_state.register_success_notice = ""
                st.markdown(
                    '<div class="auth-form-note">输入用户名和密码，继续你的对话。</div>',
                    unsafe_allow_html=True,
                )
                if st.session_state.register_saved_username or st.session_state.register_saved_email:
                    saved_username = safe_text(st.session_state.register_saved_username or "未填写")
                    saved_email = safe_text(st.session_state.register_saved_email or "未填写")
                    st.markdown(
                        f"""
                        <div class="saved-login-note">
                            已为你保留刚注册的信息。<br>
                            用户名：<strong>{saved_username}</strong><br>
                            邮箱：<strong>{saved_email}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.form("login_form", clear_on_submit=False):
                    username = st.text_input(
                        "用户名",
                        placeholder="输入你的用户名",
                        key="login_username",
                    )
                    password = st.text_input(
                        "密码",
                        type="password",
                        placeholder="输入密码",
                        key="login_password",
                    )
                    submit = st.form_submit_button("继续", use_container_width=True, type="primary")

                if submit:
                    if not username or not password:
                        st.error("请输入用户名和密码后再继续。")
                    else:
                        try:
                            response = request_json(
                                "POST",
                                "/api/login",
                                payload={"username": username, "password": password},
                                timeout=15,
                            )
                            if response.status_code == 200:
                                data = response.json()
                                st.session_state.logged_in = True
                                st.session_state.username = data.get("username", username)
                                st.session_state.access_token = data.get("access_token", "")
                                st.session_state.active_role = data.get("preferred_role", DEFAULT_ROLE_ID)
                                st.session_state.bootstrap_needed = True
                                bootstrap_user_data()
                                st.rerun()
                            else:
                                st.error(get_error_message(response))
                        except requests.RequestException as exc:
                            st.error(f"暂时无法连接服务，当前无法登录。报错信息：{exc}")

                login_saved_copy = (
                    "系统已保留最近注册时填写的账号信息，登录时可以少做一次重复输入。"
                    if st.session_state.register_saved_username or st.session_state.register_saved_email
                    else "登录成功后会延续你最近使用的角色和聊天进度。"
                )
                st.markdown(
                    f"""
                    <div class="auth-side-panel">
                        <div class="auth-side-panel-header">
                            <div class="auth-side-panel-title">登录后将恢复这些内容</div>
                            <div class="auth-side-pill">更省事</div>
                        </div>
                        <p class="auth-side-panel-copy">
                            系统会优先恢复你的常用角色、聊天入口和最近使用状态，减少重复设置，
                            让你更快回到正在进行的对话里。
                        </p>
                        <div class="auth-side-grid">
                            <div class="auth-side-card">
                                <div class="auth-side-card-label">会话延续</div>
                                <div class="auth-side-card-title">角色、节奏和聊天入口会被保留</div>
                                <div class="auth-side-card-copy">登录后会优先带你回到熟悉的使用状态，不需要从零开始。</div>
                            </div>
                            <div class="auth-side-card">
                                <div class="auth-side-card-label">输入更少</div>
                                <div class="auth-side-card-title">账号信息会尽量沿用</div>
                                <div class="auth-side-card-copy">{login_saved_copy}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    '<div class="auth-form-note">填写邮箱、用户名和密码后即可创建账号。</div>',
                    unsafe_allow_html=True,
                )
                email = st.text_input(
                    "邮箱",
                    placeholder="请输入常用邮箱，例如 name@example.com",
                    key="register_email",
                )
                email_hint_type, email_hint_message = get_email_hint(email)
                st.markdown(
                    f'<div class="email-hint {email_hint_type}">{email_hint_message}</div>',
                    unsafe_allow_html=True,
                )
                username = st.text_input("用户名", placeholder="设置一个你想长期使用的名字", key="register_username")
                show_password = st.checkbox("显示密码", key="register_show_password")
                password_input_type = "default" if show_password else "password"
                password = st.text_input(
                    "新密码",
                    type=password_input_type,
                    placeholder="8 到 20 位，包含大小写字母和数字",
                    key="register_password",
                )
                confirm_password = st.text_input(
                    "确认密码",
                    type=password_input_type,
                    placeholder="再次输入密码",
                    key="register_confirm_password",
                )

                password_strength = get_password_strength(password)
                strength_segments = "".join(
                    [
                        f'<div class="password-strength-segment {"active-" + str(index + 1) if index < password_strength["score"] else ""}"></div>'
                        for index in range(4)
                    ]
                )
                password_checks = "".join(
                    [
                        f'<div class="password-check {"pass" if passed else ""}">{"✓" if passed else "○"} {label}</div>'
                        for label, passed in password_strength["checks"]
                    ]
                )
                st.markdown(
                    f"""
                    <div class="password-strength {password_strength['class_name']}">
                        <div class="password-strength-header">
                            <div class="password-strength-label">密码强度</div>
                            <div class="password-strength-value">{password_strength['label']}</div>
                        </div>
                        <div class="password-strength-bar">{strength_segments}</div>
                        <div class="password-checklist">{password_checks}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("查看《用户协议》与《隐私政策》摘要", expanded=False):
                    st.markdown(
                        """
                        1. 请确保邮箱和用户名真实可用，方便账号识别与后续登录。
                        2. 聊天内容仅用于当前产品体验，请避免主动提交高敏感个人信息。
                        3. 涉及健康、法律、财务等高风险问题时，系统回复仅供参考，不能替代专业意见。
                        4. 为保障服务稳定，异常或违规使用行为可能会被限制访问。
                        """
                    )

                agree_terms = st.checkbox(
                    "我已阅读并同意《用户协议》与《隐私政策》",
                    key="register_agree_terms",
                )
                st.markdown(
                    '<div class="terms-note">勾选后才能继续创建账户。</div>',
                    unsafe_allow_html=True,
                )
                submit = st.button(
                    "创建账户",
                    use_container_width=True,
                    key="register_submit",
                    disabled=not agree_terms,
                )

                if submit:
                    if not email or not username or not password or not confirm_password:
                        st.error("请完整填写邮箱、用户名和密码。")
                    elif not is_valid_email(email):
                        st.error("邮箱格式不正确，请重新检查。")
                    elif password != confirm_password:
                        st.error("两次输入的密码不一致。")
                    elif (password_error := validate_password_rule(password)) is not None:
                        st.error(password_error)
                    elif not agree_terms:
                        st.error("请先勾选《用户协议》与《隐私政策》。")
                    else:
                        try:
                            response = request_json(
                                "POST",
                                "/api/register",
                                payload={"email": email, "username": username, "password": password},
                                timeout=15,
                            )
                            if response.status_code == 200:
                                normalized_email = email.strip().lower()
                                st.session_state.register_saved_username = username
                                st.session_state.register_saved_email = normalized_email
                                st.session_state.pending_login_prefill = True
                                st.session_state.pending_register_reset = True
                                st.session_state.register_success_notice = "账户已创建完成，系统已自动切换到登录并保留你的账号信息。"
                                st.rerun()
                            else:
                                st.error(get_error_message(response))
                        except requests.RequestException as exc:
                            st.error(f"暂时无法连接服务，当前无法创建账户。报错信息：{exc}")

        if ADMIN_ENABLED:
            st.write("")
            with st.expander("管理员后台入口", expanded=False):
                st.caption("这里用于查看注册用户信息和用户提问记录。请妥善保管管理员令牌。")
                admin_token_input = st.text_input(
                    "管理员令牌",
                    type="password",
                    placeholder="输入 ROLEPLAY_ADMIN_TOKEN",
                    key="admin_token_entry",
                )
                if st.button("进入管理后台", key="admin_login_submit", use_container_width=True):
                    if not admin_token_input.strip():
                        st.error("请先输入管理员令牌。")
                    else:
                        try:
                            response = request_json(
                                "GET",
                                "/api/admin/summary",
                                timeout=10,
                                extra_headers={"X-Admin-Token": admin_token_input.strip()},
                            )
                            if response.status_code == 200:
                                st.session_state.admin_logged_in = True
                                st.session_state.admin_token = admin_token_input.strip()
                                st.rerun()
                            else:
                                st.error(get_error_message(response))
                        except requests.RequestException as exc:
                            st.error(f"暂时无法连接服务，当前无法进入管理后台。报错信息：{exc}")


def show_admin_page():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Admin Console</div>
            <div class="hero-title">管理后台</div>
            <p class="hero-body">这里可以查看用户注册信息，以及用户提交过的提问/搜索内容。由于涉及敏感数据，请仅限管理员使用。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        summary_response = admin_request("/api/admin/summary", timeout=10)
    except requests.RequestException as exc:
        st.error(f"暂时无法连接管理接口。报错信息：{exc}")
        if st.button("退出管理后台", key="admin_exit_on_error"):
            clear_admin_session()
            st.rerun()
        return

    if summary_response.status_code != 200:
        message = get_error_message(summary_response)
        clear_admin_session()
        st.error(message)
        if st.button("返回登录页", key="admin_back_to_login"):
            st.rerun()
        return

    summary = summary_response.json()
    action_col, spacer_col = st.columns([0.2, 0.8], gap="medium")
    with action_col:
        if st.button("退出管理后台", key="admin_logout", use_container_width=True):
            clear_admin_session()
            st.rerun()

    summary_cards = [
        ("注册用户", f"{summary.get('user_count', 0)} 人", f"最近注册：{format_summary_time(summary.get('latest_user_created_at'))}"),
        ("用户提问", f"{summary.get('query_count', 0)} 条", f"最近记录：{format_summary_time(summary.get('latest_query_created_at'))}"),
        ("消息总量", f"{summary.get('message_count', 0)} 条", "包含用户提问和 AI 回复。"),
    ]
    metric_cols = st.columns(3, gap="medium")
    for col, (label, value, note) in zip(metric_cols, summary_cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">注册用户</div>
            <div class="section-note">支持按用户名或邮箱搜索。这里不会展示密码或密码哈希，仅展示管理所需字段。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    user_filter_col, user_limit_col = st.columns([0.75, 0.25], gap="medium")
    with user_filter_col:
        admin_user_keyword = st.text_input(
            "搜索用户",
            value=st.session_state.get("admin_user_keyword", ""),
            placeholder="输入用户名或邮箱关键词",
            key="admin_user_keyword",
        )
    with user_limit_col:
        admin_user_limit = st.number_input("用户条数", min_value=10, max_value=500, value=100, step=10, key="admin_user_limit")

    try:
        users_response = admin_request(
            "/api/admin/users",
            params={"keyword": admin_user_keyword.strip(), "limit": int(admin_user_limit)},
            timeout=12,
        )
        if users_response.status_code == 200:
            user_items = users_response.json().get("items", [])
            if user_items:
                st.dataframe(user_items, use_container_width=True, hide_index=True)
            else:
                st.info("当前没有符合条件的注册用户。")
        else:
            st.error(get_error_message(users_response))
    except requests.RequestException as exc:
        st.error(f"读取用户列表失败：{exc}")

    st.write("")
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">用户提问 / 搜索内容</div>
            <div class="section-note">这里展示的是用户发送给角色的原始输入内容。你可以按用户名、角色或关键词筛选。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    query_filter_col, query_user_col, query_role_col, query_limit_col = st.columns([0.38, 0.24, 0.2, 0.18], gap="medium")
    with query_filter_col:
        admin_query_keyword = st.text_input(
            "内容关键词",
            value=st.session_state.get("admin_query_keyword", ""),
            placeholder="搜索提问内容",
            key="admin_query_keyword",
        )
    with query_user_col:
        admin_query_user = st.text_input(
            "指定用户",
            value=st.session_state.get("admin_query_user", ""),
            placeholder="输入用户名",
            key="admin_query_user",
        )
    with query_role_col:
        role_options = ["全部角色"] + [role["name"] for role in ROLE_CONFIG.values()]
        selected_role_name = st.selectbox("角色", options=role_options, key="admin_query_role_name")
        selected_role_id = ""
        if selected_role_name != "全部角色":
            for role_id, role in ROLE_CONFIG.items():
                if role["name"] == selected_role_name:
                    selected_role_id = role_id
                    break
    with query_limit_col:
        admin_query_limit = st.number_input("记录条数", min_value=10, max_value=500, value=100, step=10, key="admin_query_limit")

    try:
        queries_response = admin_request(
            "/api/admin/queries",
            params={
                "keyword": admin_query_keyword.strip(),
                "username": admin_query_user.strip(),
                "role_id": selected_role_id,
                "limit": int(admin_query_limit),
            },
            timeout=12,
        )
        if queries_response.status_code == 200:
            query_items = queries_response.json().get("items", [])
            if query_items:
                st.dataframe(query_items, use_container_width=True, hide_index=True)
            else:
                st.info("当前没有符合条件的用户提问记录。")
        else:
            st.error(get_error_message(queries_response))
    except requests.RequestException as exc:
        st.error(f"读取用户提问记录失败：{exc}")


def render_sidebar(online):
    role_id, role = get_current_role()
    history = ensure_history(role_id)
    _, status_class = STATUS_LABELS[online]
    connection_copy = "服务连接正常，可以直接开始对话。" if online else "界面仍可查看，但发送消息时会提醒你服务未连接。"
    display_status = "连接正常" if online else "等待连接"
    message_count = max(len(history) - 1, 0)
    conversation_summaries = st.session_state.get("conversation_summaries", {})
    safe_username = safe_text(st.session_state.username)
    safe_email = safe_text(st.session_state.user_email or "未同步")
    recent_cards_markup = ""
    for item_role_id, item_role in ROLE_CONFIG.items():
        summary = conversation_summaries.get(item_role_id, {})
        last_message = summary.get("last_message") or item_role["intro"]
        preview = (last_message[:34] + "...") if len(last_message) > 34 else last_message
        updated_at = summary.get("updated_at") or "尚未开始"
        recent_cards_markup += (
            f'<div class="sidebar-stat-card{" full" if item_role_id == role_id else ""}">'
            f'<div class="sidebar-card-label">{safe_text(item_role["name"])}</div>'
            f'<div class="sidebar-card-value">{summary.get("message_count", 0)} 条</div>'
            f'<div class="sidebar-card-copy">{safe_text(preview)}</div>'
            f'<div class="sidebar-card-copy">最近更新：{safe_text(updated_at)}</div>'
            "</div>"
        )

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-hero-card">
                <div class="sidebar-kicker">Sidebar</div>
                <div class="sidebar-title">把当前会话的重点，放在一眼就能看到的位置</div>
                <p class="sidebar-copy">角色、账号、连接状态和对话进度都集中在这里，信息更少打扰，查找也更直接。</p>
                <div class="status-chip {status_class}">{display_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="sidebar-panel">
                <div class="sidebar-panel-title">当前会话</div>
                <div class="sidebar-info-grid">
                    <div class="sidebar-info-card">
                        <div class="sidebar-card-label">当前账号</div>
                        <div class="sidebar-card-value">{safe_username}</div>
                    </div>
                    <div class="sidebar-info-card">
                        <div class="sidebar-card-label">登录邮箱</div>
                        <div class="sidebar-card-value">{safe_email}</div>
                    </div>
                    <div class="sidebar-info-card">
                        <div class="sidebar-card-label">当前角色</div>
                        <div class="sidebar-card-value">{role['name']}</div>
                    </div>
                    <div class="sidebar-info-card full">
                        <div class="sidebar-card-label">回应风格</div>
                        <div class="sidebar-card-value">{role["style"]}</div>
                        <div class="sidebar-card-copy">{role["tagline"]}</div>
                    </div>
                </div>
            </div>
            <div class="sidebar-panel">
                <div class="sidebar-panel-title">会话进度</div>
                <div class="sidebar-stat-grid">
                    <div class="sidebar-stat-card">
                        <div class="sidebar-card-label">消息轮次</div>
                        <div class="sidebar-card-value">{message_count}</div>
                        <div class="sidebar-card-copy">数字越高，通常表示这段对话已经进入更深入的阶段。</div>
                    </div>
                    <div class="sidebar-stat-card">
                        <div class="sidebar-card-label">角色会话数</div>
                        <div class="sidebar-card-value">{len(st.session_state.chat_sessions)}</div>
                        <div class="sidebar-card-copy">每个角色都会保留自己的聊天上下文，不会互相混在一起。</div>
                    </div>
                    <div class="sidebar-stat-card full">
                        <div class="sidebar-card-label">服务连接</div>
                        <div class="sidebar-card-value">{display_status}</div>
                        <div class="sidebar-card-copy">{connection_copy}</div>
                    </div>
                </div>
            </div>
            <div class="sidebar-panel">
                <div class="sidebar-panel-title">最近会话</div>
                <div class="sidebar-stat-grid">{recent_cards_markup}</div>
            </div>
            <div class="sidebar-tip-card">
                <div class="sidebar-panel-title"><span class="sidebar-tip-emoji">💬</span>建议起点</div>
                <div class="sidebar-card-copy">如果你还没想好怎么开口，可以先从这句开始：{role["prompts"][0]}</div>
                <div class="sidebar-card-copy">服务地址：<a href="{BACKEND_URL}" target="_blank">{BACKEND_URL}</a></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        if st.button("清除当前对话", use_container_width=True):
            clear_current_history()
            st.rerun()
        if st.button("退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.user_email = ""
            st.session_state.access_token = ""
            st.session_state.chat_sessions = {}
            st.session_state.conversation_summaries = {}
            st.session_state.active_role = DEFAULT_ROLE_ID
            st.session_state.bootstrap_needed = False
            st.rerun()


def render_top_panel(online):
    role_id, role = get_current_role()
    status_text, status_class = STATUS_LABELS[online]
    message_count = max(len(ensure_history(role_id)) - 1, 0)

    if message_count == 0:
        stage_title = "从一句简短开场开始就很好"
        stage_copy = "先描述近况、提出一个问题，或者说出当下感受，对话就会自然展开。"
    elif message_count < 4:
        stage_title = "对话已经建立，可以继续深入"
        stage_copy = "这个阶段适合顺着上一句继续追问半步，通常很快就能进入真正想聊的核心。"
    else:
        stage_title = "上下文已经足够，可以直接进入重点"
        stage_copy = "现在适合把具体困惑、细节和真实感受都展开，对方也更容易给出更完整的回应。"

    hero_col, stats_col = st.columns([1.45, 0.9], gap="large")

    with hero_col:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="hero-kicker">Current Session</div>
                <div class="hero-title">{role['emoji']} {role['name']}</div>
                <p class="hero-body">{role['tagline']}</p>
                <div class="hero-meta-grid">
                    <div class="hero-meta-card">
                        <div class="hero-meta-label">回应风格</div>
                        <div class="hero-meta-value">{role['style']}</div>
                    </div>
                    <div class="hero-meta-card">
                        <div class="hero-meta-label">消息轮次</div>
                        <div class="hero-meta-value">{message_count} 轮</div>
                    </div>
                    <div class="hero-meta-card">
                        <div class="hero-meta-label">服务状态</div>
                        <div class="hero-meta-value">{status_text}</div>
                    </div>
                </div>
                <div class="hero-detail-grid">
                    <div class="hero-detail-card">
                        <div class="hero-detail-label">适用场景</div>
                        <div class="hero-detail-title">这种状态会更适合它</div>
                        <div class="hero-detail-copy">{role['panel_fit']}</div>
                    </div>
                    <div class="hero-detail-card">
                        <div class="hero-detail-label">建议起点</div>
                        <div class="hero-detail-title">不知道怎么开口时</div>
                        <div class="hero-detail-copy">{role['panel_opening']}</div>
                    </div>
                    <div class="hero-detail-card">
                        <div class="hero-detail-label">当前阶段</div>
                        <div class="hero-detail-title">{stage_title}</div>
                        <div class="hero-detail-copy">{stage_copy}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with stats_col:
        stat_cards = [
            ("服务状态", status_text, "连接正常时可以直接发送消息；若服务暂不可用，界面也会明确提醒。"),
            ("当前账号", safe_text(st.session_state.username), "你的账号会保留当前使用习惯，角色只负责回应与你协作。"),
        ]
        stat_cards_markup = "".join(
            [
                f'<div class="top-stat-card"><div class="top-stat-label">{label}</div><div class="top-stat-value">{value}</div><div class="top-stat-copy">{note}</div></div>'
                for label, value, note in stat_cards
            ]
        )
        st.markdown(f'<div class="top-stat-grid">{stat_cards_markup}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title">推荐使用方式</div>
                <div class="section-note">先确认当前角色，再选一个合适的起点，随后直接进入聊天。整个流程尽量保持简洁，不打断你的思路。</div>
                <div class="status-chip {status_class}">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_role_picker():
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">选择一个更适合当前状态的角色</div>
            <div class="section-note">角色切换不仅会改变名称，也会一起改变回应方式、语气和建议节奏。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    items = list(ROLE_CONFIG.items())
    for start in range(0, len(items), 3):
        row_items = items[start:start + 3]
        row_length = len(row_items)
        if row_length == 1:
            _, center_col, _ = st.columns([0.55, 1, 0.55], gap="medium")
            target_columns = [center_col]
        elif row_length == 2:
            _, left_col, right_col, _ = st.columns([0.18, 1, 1, 0.18], gap="medium")
            target_columns = [left_col, right_col]
        else:
            target_columns = st.columns(3, gap="medium")

        for col, role_item in zip(target_columns, row_items):
            role_id, role = role_item
            active_class = "active" if role_id == st.session_state.active_role else ""
            with col:
                st.markdown('<div class="role-card-anchor"></div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="role-card {active_class}">
                        <div class="role-emoji">{role['emoji']}</div>
                        <div class="role-name">{role['name']}</div>
                        <div class="role-tagline">{role['tagline']}</div>
                        <div class="role-style">风格特点：{role['style']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                button_type = "primary" if role_id == st.session_state.active_role else "secondary"
                if st.button(
                    "当前使用中" if role_id == st.session_state.active_role else "切换到此角色",
                    key=f"switch_{role_id}",
                    use_container_width=True,
                    type=button_type,
                ):
                    st.session_state.active_role = role_id
                    ensure_history(role_id)
                    if st.session_state.logged_in:
                        try:
                            request_json(
                                "POST",
                                "/api/preferences/active-role",
                                payload={"role_id": role_id},
                                timeout=10,
                                auth_required=True,
                            )
                        except requests.RequestException:
                            pass
                    st.rerun()


def render_prompt_panel():
    _, role = get_current_role()
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">如果你想更快开始，可以直接使用这些建议起点</div>
            <div class="section-note">它们不是固定模板，而是帮助你更自然进入对话的第一句。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(len(role["prompts"]), gap="medium")
    for col, prompt in zip(columns, role["prompts"]):
        with col:
            st.markdown(f'<div class="quick-prompt">{prompt}</div>', unsafe_allow_html=True)
            if st.button("使用这句", key=f"prompt_{prompt}", use_container_width=True):
                submit_message(prompt)
                st.rerun()


def render_knowledge_upload_panel(compact=False):
    if compact:
        st.markdown(
            """
            <div class="knowledge-upload-compact">
                <div class="knowledge-upload-title">添加参考文件</div>
                <p class="knowledge-upload-copy">上传 PDF 或图片后会先解析并加入知识库，再继续在下方输入问题。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">上传知识文件</div>
                <div class="section-note">支持 PDF 和图片。上传后会自动解析文字内容并写入当前知识库；后续对话会从这些内容中检索参考信息。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader(
        "选择 PDF 或图片文件",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        accept_multiple_files=False,
        key="knowledge_file_uploader_compact" if compact else "knowledge_file_uploader",
        label_visibility="collapsed" if compact else "visible",
    )

    if compact:
        upload_clicked = st.button(
            "解析并加入知识库",
            use_container_width=True,
            type="primary",
            disabled=uploaded_file is None,
            key="knowledge_file_upload_submit_compact" if compact else "knowledge_file_upload_submit",
        )
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.caption(f"已选择：{uploaded_file.name}，大小 {file_size_mb:.2f} MB。")
        else:
            st.caption("PDF 支持复杂版面和表格；图片支持 OCR 识别文字内容。")
    else:
        action_col, info_col = st.columns([0.32, 0.68], gap="medium")
        with action_col:
            upload_clicked = st.button(
                "解析并加入知识库",
                use_container_width=True,
                type="primary",
                disabled=uploaded_file is None,
                key="knowledge_file_upload_submit",
            )
        with info_col:
            if uploaded_file is not None:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.caption(f"已选择：{uploaded_file.name}，大小 {file_size_mb:.2f} MB。")
            else:
                st.caption("PDF 支持复杂版面和表格；图片支持 OCR 识别文字内容。")

    if upload_clicked and uploaded_file is not None:
        extension = os.path.splitext(uploaded_file.name)[1].lower()
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        if extension == ".pdf":
            endpoint = "/api/knowledge/upload-pdf"
            mime_type = "application/pdf"
            file_label = "PDF"
        elif extension in image_extensions:
            endpoint = "/api/knowledge/upload-image"
            mime_type = uploaded_file.type or "application/octet-stream"
            file_label = "图片"
        else:
            st.error("只支持上传 PDF 或常见图片格式。")
            return

        try:
            with st.spinner(f"正在解析{file_label}并写入知识库..."):
                response = upload_knowledge_file_request(uploaded_file, endpoint, mime_type)
            if response.status_code == 200:
                data = response.json()
                page_copy = f"{data.get('pages', 0)} 页，" if data.get("pages") is not None else ""
                st.success(
                    f"{file_label}已入库：{data.get('filename', uploaded_file.name)}；"
                    f"解析器 {data.get('parser', 'unknown')}，"
                    f"{page_copy}{data.get('chunks', 0)} 个文本块。"
                )
                warnings = data.get("warnings") or []
                if warnings:
                    with st.expander("解析提示", expanded=False):
                        for warning in warnings:
                            st.warning(warning)
            else:
                st.error(get_error_message(response))
        except requests.RequestException as exc:
            st.error(f"文件上传或解析失败：{exc}")


def render_chat_panel():
    role_id, role = get_current_role()
    history = ensure_history(role_id)

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">对话区</div>
            <div class="section-note">这里会保留你与当前角色的真实聊天上下文。你可以从一句简短开场开始，也可以直接进入重点。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

    if not history:
        st.markdown(
            """
            <div class="empty-state">
                这里还没有内容。发送一句近况、一个问题，或一段你此刻的想法，对话就会开始。
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for message in history:
            avatar = role["emoji"] if message["role"] == "assistant" else "🧑"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    st.markdown("</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="chat-composer-anchor"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="chat-composer-header">
                <div class="chat-composer-kicker">输入区</div>
                <div class="chat-composer-title">继续和 {role['name']} 对话</div>
                <p class="chat-composer-copy">把问题、背景或目标直接写进来，当前角色会沿着你的语境继续回应。</p>
                <div class="chat-composer-pill">{role['emoji']} 当前角色</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _, composer_col, _ = st.columns([0.035, 0.93, 0.035], gap="small")
        with composer_col:
            render_knowledge_upload_panel(compact=True)
            with st.form("chat_compose_form", clear_on_submit=True):
                prompt = st.text_area(
                    "消息输入",
                    placeholder=f"输入你想对 {role['name']} 说的话",
                    key="chat_message_input",
                    height=110,
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("发送消息", use_container_width=True, type="primary")

        if submitted:
            if prompt.strip():
                submit_message(prompt)
                st.rerun()
            else:
                st.warning("先写点内容，再发送。")


def show_chat_page():
    if st.session_state.bootstrap_needed or not st.session_state.chat_sessions:
        if not bootstrap_user_data():
            st.rerun()
            return
    online, _ = get_backend_status(BACKEND_URL)
    render_sidebar(online)
    render_top_panel(online)
    st.write("")
    render_role_picker()
    st.write("")
    render_prompt_panel()
    st.write("")
    render_chat_panel()
    st.markdown(
        '<div class="footer-note">如果服务暂不可用，界面会在需要发送时直接给出明确提示，避免无效操作。</div>',
        unsafe_allow_html=True,
    )


init_session_state()
inject_styles()
ensure_history(st.session_state.active_role)

if st.session_state.admin_logged_in:
    show_admin_page()
elif st.session_state.logged_in:
    show_chat_page()
else:
    show_login_page()
