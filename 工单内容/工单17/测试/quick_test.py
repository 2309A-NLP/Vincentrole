#!/usr/bin/env python3
"""
RAGFlow API 快速验证脚本
用于验证API是否可用，并获取基准性能数据
"""

import requests
import time
import json
import sys

BASE_URL = "http://localhost:9380"

def test_health():
    """测试系统健康状态"""
    print("1. 测试系统健康状态...")
    try:
        # 测试Web界面
        resp = requests.get("http://localhost:8080/", timeout=5)
        print(f"   Web界面: ✓ (HTTP {resp.status_code})")
        
        # 测试API（可能需要认证）
        resp = requests.get(f"{BASE_URL}/api/v1/system", timeout=5)
        print(f"   API服务: ✓ (HTTP {resp.status_code})")
        
        return True
    except Exception as e:
        print(f"   错误: {e}")
        return False

def test_chat_api(api_key: str, chat_id: str):
    """测试聊天API"""
    print("\n2. 测试聊天API...")
    
    url = f"{BASE_URL}/api/v1/chats_openai/{chat_id}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": "什么是RAG？"}],
        "stream": False
    }
    
    try:
        start = time.time()
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        latency = (time.time() - start) * 1000
        
        data = resp.json()
        success = resp.status_code == 200 and data.get("code") == 0
        
        print(f"   状态码: {resp.status_code}")
        print(f"   响应时间: {latency:.2f}ms")
        print(f"   成功: {'✓' if success else '✗'}")
        
        if success:
            answer = data.get("data", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens = data.get("data", {}).get("usage", {}).get("total_tokens", 0)
            print(f"   Token使用: {tokens}")
            print(f"   回答预览: {answer[:100]}...")
        
        return success, latency
    except Exception as e:
        print(f"   错误: {e}")
        return False, 0

def run_baseline_test(api_key: str, chat_id: str, num_requests: int = 5):
    """运行基准测试"""
    print(f"\n3. 运行基准测试 ({num_requests}次请求)...")
    
    questions = [
        "什么是RAG？",
        "RAG系统的主要组件有哪些？",
        "文档解析的作用是什么？",
        "向量检索的原理是什么？",
        "如何优化RAG系统的性能？"
    ]
    
    latencies = []
    successes = 0
    
    for i in range(num_requests):
        question = questions[i % len(questions)]
        url = f"{BASE_URL}/api/v1/chats_openai/{chat_id}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "model",
            "messages": [{"role": "user", "content": question}],
            "stream": False
        }
        
        try:
            start = time.time()
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            latency = (time.time() - start) * 1000
            
            data = resp.json()
            success = resp.status_code == 200 and data.get("code") == 0
            
            if success:
                successes += 1
            latencies.append(latency)
            
            print(f"   请求 {i+1}: {latency:.2f}ms {'✓' if success else '✗'}")
            
            # 模拟用户思考时间
            time.sleep(1)
            
        except Exception as e:
            print(f"   请求 {i+1}: 错误 - {e}")
            latencies.append(0)
    
    # 计算统计
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"\n=== 基准测试结果 ===")
        print(f"成功率: {successes}/{num_requests} ({successes/num_requests*100:.1f}%)")
        print(f"平均响应时间: {avg_latency:.2f}ms")
        print(f"最小响应时间: {min_latency:.2f}ms")
        print(f"最大响应时间: {max_latency:.2f}ms")
        
        return {
            "success_rate": successes/num_requests,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "latencies": latencies
        }
    
    return None

def main():
    if len(sys.argv) < 3:
        print("用法: python quick_test.py <API_KEY> <CHAT_ID>")
        print("示例: python quick_test.py ragflow-xxxxxxxxxxxx xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
        sys.exit(1)
    
    api_key = sys.argv[1]
    chat_id = sys.argv[2]
    
    print("=== RAGFlow API 快速验证 ===\n")
    
    # 测试健康状态
    if not test_health():
        print("系统健康检查失败，请检查服务状态")
        sys.exit(1)
    
    # 测试单次请求
    success, latency = test_chat_api(api_key, chat_id)
    if not success:
        print("聊天API测试失败，请检查API Key和Chat ID")
        sys.exit(1)
    
    # 运行基准测试
    result = run_baseline_test(api_key, chat_id, num_requests=5)
    
    if result:
        # 保存结果
        output_path = "/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单17/测试/baseline_test_result.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n基准测试结果已保存到: {output_path}")
    
    print("\n=== 验证完成 ===")
    print("如果一切正常，可以运行完整的压测脚本：")
    print(f"python ragflow_load_test.py --api-key {api_key} --chat-id {chat_id} --scenario both")

if __name__ == "__main__":
    main()
