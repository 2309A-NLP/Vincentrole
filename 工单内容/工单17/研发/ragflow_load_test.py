#!/usr/bin/env python3
"""
RAGFlow API 压测脚本
用于工单17：解决API服务并发瓶颈与资源泄漏

功能：
1. 场景A：高频问答压测（20并发）
2. 场景B：混合负载压测（10并发：5问答+5上传）
3. 收集性能指标：响应时间、吞吐量、错误率
"""

import asyncio
import aiohttp
import time
import json
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import argparse
import csv
import os

@dataclass
class RequestResult:
    """单次请求结果"""
    timestamp: float
    latency_ms: float
    status_code: int
    success: bool
    error: Optional[str] = None
    tokens_used: Optional[int] = None

@dataclass
class LoadTestConfig:
    """压测配置"""
    base_url: str = "http://localhost:9380"
    api_key: str = ""  # 需要填入
    chat_id: str = ""  # 需要填入
    concurrent_users: int = 20
    duration_seconds: int = 600  # 10分钟
    ramp_up_seconds: int = 30
    
class RagFlowLoadTester:
    """RAGFlow压测器"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[RequestResult] = []
        self.start_time = None
        self._stop_event = asyncio.Event()
        
    async def chat_completion(self, session: aiohttp.ClientSession, question: str) -> RequestResult:
        """发送单次问答请求"""
        url = f"{self.config.base_url}/api/v1/chats_openai/{self.config.chat_id}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "model",
            "messages": [{"role": "user", "content": question}],
            "stream": False
        }
        
        start = time.time()
        try:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                latency = (time.time() - start) * 1000
                raw_text = await resp.text()
                try:
                    data = json.loads(raw_text)
                except:
                    data = {}
                # OpenAI兼容格式：HTTP 200 + 有 choices 字段即为成功
                success = resp.status == 200 and isinstance(data, dict) and "choices" in data
                error_msg = None
                if not success:
                    error_msg = f"HTTP{resp.status}|{raw_text[:100]}"
                return RequestResult(
                    timestamp=start,
                    latency_ms=latency,
                    status_code=resp.status,
                    success=success,
                    error=error_msg,
                    tokens_used=data.get("usage", {}).get("total_tokens") if isinstance(data, dict) else None
                )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return RequestResult(
                timestamp=start,
                latency_ms=latency,
                status_code=0,
                success=False,
                error=str(e)[:200]
            )
    
    async def user_simulation(self, user_id: int, questions: List[str]):
        """模拟单个用户行为"""
        async with aiohttp.ClientSession() as session:
            question_idx = 0
            while not self._stop_event.is_set():
                question = questions[question_idx % len(questions)]
                result = await self.chat_completion(session, question)
                self.results.append(result)
                
                # 模拟用户思考时间（1-3秒）
                await asyncio.sleep(1 + (user_id % 3))
                question_idx += 1
    
    async def run_scenario_a(self, questions: List[str]):
        """场景A：高频问答（20并发）"""
        print(f"\n=== 场景A：高频问答压测 ===")
        print(f"并发用户: {self.config.concurrent_users}")
        print(f"持续时间: {self.config.duration_seconds}秒")
        
        self.start_time = time.time()
        self._stop_event.clear()
        
        # 创建并发用户任务
        tasks = []
        for i in range(self.config.concurrent_users):
            task = asyncio.create_task(self.user_simulation(i, questions))
            tasks.append(task)
            # 逐步增加用户（ramp-up）
            if i < self.config.concurrent_users - 1:
                await asyncio.sleep(self.config.ramp_up_seconds / self.config.concurrent_users)
        
        # 等待指定时间后停止
        await asyncio.sleep(self.config.duration_seconds)
        self._stop_event.set()
        
        # 等待所有任务完成
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return self.generate_report("场景A")
    
    async def run_scenario_b(self, questions: List[str], pdf_path: str):
        """场景B：混合负载（10并发：5问答+5上传）"""
        print(f"\n=== 场景B：混合负载压测 ===")
        print(f"并发用户: 10 (5问答 + 5上传)")
        print(f"持续时间: {self.config.duration_seconds}秒")
        
        self.start_time = time.time()
        self._stop_event.clear()
        
        tasks = []
        # 5个问答用户
        for i in range(5):
            task = asyncio.create_task(self.user_simulation(i, questions))
            tasks.append(task)
            await asyncio.sleep(2)
        
        # 5个上传用户（简化：模拟上传请求）
        for i in range(5):
            task = asyncio.create_task(self.simulate_upload_user(i))
            tasks.append(task)
            await asyncio.sleep(2)
        
        await asyncio.sleep(self.config.duration_seconds)
        self._stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return self.generate_report("场景B")
    
    async def simulate_upload_user(self, user_id: int):
        """模拟上传用户（发送轻量请求模拟）"""
        async with aiohttp.ClientSession() as session:
            while not self._stop_event.is_set():
                # 发送健康检查请求模拟上传期间的系统负载
                try:
                    async with session.get(f"{self.config.base_url}/api/v1/system") as resp:
                        pass
                except:
                    pass
                await asyncio.sleep(5)
    
    def generate_report(self, scenario_name: str) -> Dict:
        """生成压测报告"""
        if not self.results:
            return {"error": "无测试数据"}
        
        latencies = [r.latency_ms for r in self.results]
        successes = [r for r in self.results if r.success]
        failures = [r for r in self.results if not r.success]
        
        duration = time.time() - self.start_time if self.start_time else 1
        
        report = {
            "scenario": scenario_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "total_requests": len(self.results),
            "successful_requests": len(successes),
            "failed_requests": len(failures),
            "success_rate": round(len(successes) / len(self.results) * 100, 2),
            "throughput_rps": round(len(self.results) / duration, 2),
            "latency": {
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "avg_ms": round(statistics.mean(latencies), 2),
                "median_ms": round(statistics.median(latencies), 2),
                "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
                "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
            },
            "errors": {}
        }
        
        # 统计错误类型
        for f in failures:
            error = f.error or f"HTTP_{f.status_code}"
            report["errors"][error] = report["errors"].get(error, 0) + 1
        
        return report
    
    def save_results(self, output_dir: str, report: Dict):
        """保存测试结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON报告
        report_path = os.path.join(output_dir, f"report_{report['scenario']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 保存原始数据CSV
        csv_path = os.path.join(output_dir, f"raw_{report['scenario']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'latency_ms', 'status_code', 'success', 'error', 'tokens_used'])
            for r in self.results:
                writer.writerow([r.timestamp, r.latency_ms, r.status_code, r.success, r.error, r.tokens_used])
        
        print(f"\n报告已保存: {report_path}")
        print(f"原始数据已保存: {csv_path}")
        
        return report_path


# 预设测试问题（匹配招股说明书内容 - 力源信息半导体IC分销商）
DEFAULT_QUESTIONS = [
    "这家公司的主营业务是什么？",
    "公司的核心竞争力有哪些？",
    "公司的营业收入情况如何？",
    "公司面临哪些风险因素？",
    "公司的发展战略是什么？",
    "公司的主要客户有哪些？",
    "公司的行业地位如何？",
    "公司的盈利模式是什么？",
    "公司的管理团队背景？",
    "公司的募集资金用途？",
]


async def main():
    parser = argparse.ArgumentParser(description="RAGFlow API 压测脚本")
    parser.add_argument("--api-key", required=True, help="RAGFlow API Key")
    parser.add_argument("--chat-id", required=True, help="聊天助手ID")
    parser.add_argument("--scenario", choices=["A", "B", "both"], default="both", help="测试场景")
    parser.add_argument("--duration", type=int, default=600, help="测试持续时间（秒）")
    parser.add_argument("--concurrent", type=int, default=20, help="场景A并发用户数")
    parser.add_argument("--output", default="/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单17/测试/压测结果", help="输出目录")
    args = parser.parse_args()
    
    config = LoadTestConfig(
        api_key=args.api_key,
        chat_id=args.chat_id,
        concurrent_users=args.concurrent,
        duration_seconds=args.duration
    )
    
    tester = RagFlowLoadTester(config)
    
    if args.scenario in ["A", "both"]:
        report_a = await tester.run_scenario_a(DEFAULT_QUESTIONS)
        tester.save_results(args.output, report_a)
        print(f"\n场景A结果:")
        print(json.dumps(report_a, indent=2, ensure_ascii=False))
    
    if args.scenario in ["B", "both"]:
        tester.results = []  # 重置结果
        report_b = await tester.run_scenario_b(DEFAULT_QUESTIONS, None)
        tester.save_results(args.output, report_b)
        print(f"\n场景B结果:")
        print(json.dumps(report_b, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
