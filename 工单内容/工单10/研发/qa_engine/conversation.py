"""
工单编号: 人工智能NLP-RAG-Query理解优化任务
多轮对话管理 - 上下文维护与指代消解

核心功能:
1. 维护对话历史（最近N轮）
2. 指代消解（"他/这个公司/那"等代词解析为具体实体）
3. 公司上下文追踪（记住当前讨论的公司）
4. 生成带历史上下文的完整查询
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import deque


class ConversationManager:
    """
    多轮对话管理器
    
    设计原则:
    - 保持简单：只追踪最近N轮对话
    - 公司名是核心上下文：问答系统常在同一公司内切换话题
    - 指代消解基于规则：可靠且可解释
    """

    def __init__(self, max_history: int = 3, context_window: int = 2000):
        """
        Args:
            max_history: 保留最近几轮对话
            context_window: 历史上下文最大字符数
        """
        self.max_history = max_history
        self.context_window = context_window
        self.history: deque = deque(maxlen=max_history * 2)  # [Q, A, Q, A, ...]
        self.current_company = ""  # 当前讨论的公司
        self.last_entity = ""  # 上一轮提到的核心实体（人名/奖项等）

    def reset(self):
        """清空对话历史"""
        self.history.clear()
        self.current_company = ""
        self.last_entity = ""

    def add_turn(self, question: str, answer: str):
        """添加一轮对话"""
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        
        # 更新上下文
        self._update_context(question, answer)

    def _update_context(self, question: str, answer: str):
        """更新公司和实体上下文"""
        # 1. 尝试从问题中提取公司名
        company_from_q = self._extract_company(question)
        if company_from_q:
            self.current_company = company_from_q
        
        # 2. 如果问题中没有，尝试从回答中提取
        if not company_from_q:
            company_from_a = self._extract_company(answer)
            if company_from_a:
                self.current_company = company_from_a
        
        # 3. 提取核心实体（人名、奖项等）
        entity = self._extract_key_entity(answer)
        if entity:
            self.last_entity = entity

    def _extract_company(self, text: str) -> str:
        """从文本中提取公司名"""
        # 匹配"XX股份有限公司"、"XX有限公司"等
        patterns = [
            r'[\u4e00-\u9fa5]+(?:股份|集团|控股)?(?:有限)?公司',
            r'[\u4e00-\u9fa5]+(?:股份|集团|控股)?有限公司',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 优先选择最长的匹配（更具体）
                longest = max(matches, key=len)
                # 过滤掉一些常见的非公司名
                if len(longest) >= 4 and "公司" in longest:
                    return longest
        
        return ""

    def _extract_key_entity(self, text: str) -> str:
        """从回答中提取核心实体（人名、奖项、项目名等）"""
        # 1. 人名（2-4个中文字符，常见姓氏开头）
        person_patterns = [
            r'[张李王刘陈杨黄赵周吴徐孙马朱胡郭何林罗高梁郑谢宋唐许邓冯韩曹曾彭萧蔡潘田董袁于余叶蒋杜苏魏程吕丁沈任姚卢傅钟姜崔谭廖范汪陆金石戴贾韦夏邱方侯邹熊孟秦白江阎薛尹段雷龙史陶贺顾毛郝龚邵万钱严赖覃康洪][\u4e00-\u9fa5]{1,3}',
            r'[A-Z][a-z]+(?: [A-Z][a-z]+)?',  # 英文名
        ]
        
        for pattern in person_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 返回第一个匹配的人名
                return matches[0]
        
        # 2. 奖项名称
        award_patterns = [
            r'[\u4e00-\u9fa5]*(?:奖|奖章|荣誉|称号)[\u4e00-\u9fa5]*',
            r'国家[\u4e00-\u9fa5]*奖',
            r'[\u4e00-\u9fa5]*一等奖',
        ]
        
        for pattern in award_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        
        # 3. 工程/项目名称
        project_patterns = [
            r'[\u4e00-\u9fa5]*(?:工程|项目|系统|平台)[\u4e00-\u9fa5]*',
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 过滤太短的匹配
                valid_matches = [m for m in matches if len(m) >= 4]
                if valid_matches:
                    return valid_matches[0]
        
        return ""

    def resolve_references(self, question: str) -> Tuple[str, dict]:
        """
        解析问题中的指代词，返回消解后的问题和消解信息
        
        Returns:
            (resolved_question, resolution_info)
        """
        original = question
        resolution_info = {
            "has_resolution": False,
            "original": original,
            "resolved": question,
            "company_resolved": "",
            "entity_resolved": "",
        }
        
        # 1. 处理"这个公司"、"该公司"等
        company_patterns = [
            (r'这个公司', self.current_company),
            (r'该公司', self.current_company),
            (r'其公司', self.current_company),
            (r'上述公司', self.current_company),
        ]
        
        for pattern, company in company_patterns:
            if pattern in question and company:
                question = question.replace(pattern, company)
                resolution_info["has_resolution"] = True
                resolution_info["company_resolved"] = company
                break
        
        # 2. 处理"他"、"她"、"它"等代词
        pronoun_patterns = [
            (r'他参与', f"{self.last_entity}参与" if self.last_entity else ""),
            (r'他的', f"{self.last_entity}的" if self.last_entity else ""),
            (r'他获得了', f"{self.last_entity}获得了" if self.last_entity else ""),
            (r'他负责', f"{self.last_entity}负责" if self.last_entity else ""),
        ]
        
        for pattern, replacement in pronoun_patterns:
            if pattern in question and replacement:
                question = question.replace(pattern, replacement)
                resolution_info["has_resolution"] = True
                resolution_info["entity_resolved"] = self.last_entity
                break
        
        # 3. 处理"那XX呢"（切换公司）
        switch_pattern = r'那([\u4e00-\u9fa5]+(?:公司|集团|股份)?)(?:呢)?'
        match = re.search(switch_pattern, question)
        if match:
            new_company = match.group(1)
            if new_company and new_company != self.current_company:
                self.current_company = new_company
                resolution_info["has_resolution"] = True
                resolution_info["company_resolved"] = new_company
        
        # 4. 处理简单代词"呢"（追问同一主题）
        if question.endswith("呢？") or question.endswith("呢?"):
            # 保持当前上下文，不做替换
            pass
        
        resolution_info["resolved"] = question
        return question, resolution_info

    def get_history_context(self) -> str:
        """获取历史上下文字符串，用于注入LLM提示"""
        if not self.history:
            return ""
        
        context_parts = []
        total_len = 0
        
        # 从最近的对话开始，向前收集
        for item in reversed(self.history):
            role = "用户" if item["role"] == "user" else "助手"
            content = item["content"]
            
            # 截断过长的内容
            if len(content) > 200:
                content = content[:200] + "..."
            
            line = f"{role}: {content}"
            if total_len + len(line) > self.context_window:
                break
            
            context_parts.insert(0, line)
            total_len += len(line)
        
        if context_parts:
            return "【对话历史】\n" + "\n".join(context_parts) + "\n【以上是对话历史，请结合上下文理解当前问题】\n"
        return ""

    def get_status(self) -> dict:
        """获取当前对话状态"""
        return {
            "history_count": len(self.history) // 2,
            "current_company": self.current_company,
            "last_entity": self.last_entity,
            "max_history": self.max_history,
        }