"""工具函数"""

import re


def detect_language(text: str) -> str:
    """检测语言：'zh' 中文 / 'en' 英文"""
    if not text:
        return "zh"
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    ascii_chars = len(re.findall(r'[a-zA-Z]', text))
    total = max(chinese_chars + ascii_chars, 1)
    if chinese_chars / total >= 0.3:
        return "zh"
    return "en"
