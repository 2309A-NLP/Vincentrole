#!/usr/bin/env python3
"""
从chunk_metadata.json生成金融领域问答对数据集
用于Embedding模型微调
"""

import json
import re
import os
from typing import List, Dict, Tuple

# 配置
CHUNK_METADATA_PATH = "/Users/suwente/Desktop/专高六学习资料/RAG 工单/附件/ccf_competition 2/chunk_metadata.json"
OUTPUT_PATH = "./training_data.json"

# 公司名称映射
COMPANY_MAP = {
    "中国平安保险集团股份有限公司": "中国平安",
    "中国太平洋保险集团股份有限公司": "中国太保",
    "宝山钢铁股份有限公司": "宝钢股份",
    "重庆涪陵榨菜集团股份有限公司": "涪陵榨菜",
    "三一重工股份有限公司": "三一重工",
    "中国南方航空股份有限公司": "南方航空",
    "美的集团股份有限公司": "美的集团",
    "双汇发展股份有限公司": "双汇发展",
    "佛山市海天调味食品股份有限公司": "海天味业",
    "泸州老窖股份有限公司": "泸州老窖"
}

# 财务关键词
FINANCIAL_KEYWORDS = [
    "营业收入", "净利润", "总资产", "总负债", "股东权益", 
    "每股收益", "净资产收益率", "毛利率", "净利率",
    "经营活动现金流量", "投资活动现金流量", "筹资活动现金流量",
    "资产负债率", "流动比率", "速动比率"
]

def extract_company_year(source_file: str) -> Tuple[str, str]:
    """从文件名提取公司名和年份"""
    # 格式: 2020-02-21__中国平安保险集团股份有限公司__601318__中国平安__2019年__年度报告.pdf
    parts = source_file.split("__")
    if len(parts) >= 5:
        company_full = parts[1]
        year_part = parts[4]
        # 提取年份
        year_match = re.search(r"(\d{4})年", year_part)
        year = year_match.group(1) if year_match else ""
        
        # 获取简称
        company_short = COMPANY_MAP.get(company_full, company_full)
        return company_short, year
    return "", ""

def extract_financial_data(text: str, keyword: str) -> List[str]:
    """从文本中提取财务数据"""
    # 简单模式匹配：关键词后跟数字
    patterns = [
        rf"{keyword}[^0-9]*?([\d,]+\.?\d*)\s*(?:亿元|元|%)",
        rf"{keyword}[^0-9]*?([\d,]+\.?\d*)",
    ]
    
    results = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # 清理数字格式
            clean_num = match.replace(",", "")
            if clean_num and float(clean_num) > 0:
                results.append(clean_num)
    return results

def generate_qa_pairs(chunks: List[Dict]) -> List[Dict]:
    """生成问答对"""
    qa_pairs = []
    
    for chunk in chunks:
        text = chunk["text"]
        source_file = chunk["source_file"]
        
        # 提取公司名和年份
        company, year = extract_company_year(source_file)
        if not company or not year:
            continue
        
        # 检查是否包含财务关键词
        for keyword in FINANCIAL_KEYWORDS:
            if keyword in text:
                # 提取数据
                values = extract_financial_data(text, keyword)
                if values:
                    # 生成问题
                    question_templates = [
                        f"{company}{year}年的{keyword}是多少？",
                        f"{company}在{year}年的{keyword}是什么？",
                        f"请问{company}{year}年的{keyword}？",
                        f"{year}年{company}的{keyword}是多少？"
                    ]
                    
                    # 使用第一个值作为答案
                    answer = values[0]
                    
                    # 添加单位
                    if keyword in ["营业收入", "净利润", "总资产", "总负债", "股东权益", 
                                 "经营活动现金流量", "投资活动现金流量", "筹资活动现金流量"]:
                        answer = f"{answer}亿元"
                    elif keyword in ["每股收益"]:
                        answer = f"{answer}元"
                    elif keyword in ["净资产收益率", "毛利率", "净利率", "资产负债率"]:
                        answer = f"{answer}%"
                    
                    # 随机选择问题模板
                    import random
                    question = random.choice(question_templates)
                    
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "context": text[:500],  # 上下文片段
                        "company": company,
                        "year": year,
                        "keyword": keyword,
                        "source": source_file
                    })
    
    return qa_pairs

def main():
    """主函数"""
    print("加载chunk_metadata.json...")
    with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data["chunks"]
    print(f"总chunks数: {len(chunks)}")
    
    print("生成问答对...")
    qa_pairs = generate_qa_pairs(chunks)
    print(f"生成问答对数量: {len(qa_pairs)}")
    
    # 统计信息
    companies = set(qa["company"] for qa in qa_pairs)
    keywords = set(qa["keyword"] for qa in qa_pairs)
    print(f"涉及公司: {len(companies)}家")
    print(f"涉及财务指标: {len(keywords)}个")
    
    # 保存结果
    print(f"保存到: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    # 显示示例
    print("\n示例问答对:")
    for i, qa in enumerate(qa_pairs[:5]):
        print(f"{i+1}. 问题: {qa['question']}")
        print(f"   答案: {qa['answer']}")
        print(f"   公司: {qa['company']}, 年份: {qa['year']}")
        print()

if __name__ == "__main__":
    main()