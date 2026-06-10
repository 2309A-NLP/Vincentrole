#!/usr/bin/env python3
"""
Docker容器资源监控脚本
用于工单17：监控RAGFlow服务的资源使用情况

监控指标：
1. 容器CPU使用率
2. 容器内存使用量和使用率
3. 容器网络I/O
4. 容器磁盘I/O
"""

import subprocess
import json
import time
import csv
import os
from datetime import datetime
from typing import Dict, List
import argparse

class DockerResourceMonitor:
    """Docker资源监控器"""
    
    def __init__(self, container_names: List[str], output_dir: str):
        self.container_names = container_names
        self.output_dir = output_dir
        self.monitoring = False
        self.stats_history: List[Dict] = []
        
    def get_container_stats(self, container_name: str) -> Dict:
        """获取单个容器的资源统计"""
        try:
            cmd = f"docker stats {container_name} --no-stream --format '{{{{json .}}}}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                stats = json.loads(result.stdout)
                return {
                    "container": container_name,
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": stats.get("CPUPerc", "0%").strip('%'),
                    "mem_usage": stats.get("MemUsage", "0B / 0B"),
                    "mem_percent": stats.get("MemPerc", "0%").strip('%'),
                    "net_io": stats.get("NetIO", "0B / 0B"),
                    "block_io": stats.get("BlockIO", "0B / 0B"),
                    "pids": stats.get("PIDs", "0")
                }
        except Exception as e:
            print(f"获取容器 {container_name} 统计失败: {e}")
        return None
    
    def get_all_stats(self) -> List[Dict]:
        """获取所有监控容器的统计"""
        stats = []
        for name in self.container_names:
            stat = self.get_container_stats(name)
            if stat:
                stats.append(stat)
        return stats
    
    def monitor(self, interval: int = 5, duration: int = 600):
        """持续监控资源使用"""
        print(f"开始监控Docker容器资源...")
        print(f"监控容器: {', '.join(self.container_names)}")
        print(f"采样间隔: {interval}秒")
        print(f"监控时长: {duration}秒")
        
        self.monitoring = True
        start_time = time.time()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # CSV文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(self.output_dir, f"resource_monitor_{timestamp}.csv")
        
        # 写入CSV头
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'container', 'cpu_percent', 
                'mem_usage', 'mem_percent', 'net_io', 'block_io', 'pids'
            ])
        
        print(f"监控数据将保存到: {csv_path}")
        print("按 Ctrl+C 停止监控\n")
        
        try:
            while self.monitoring and (time.time() - start_time) < duration:
                stats = self.get_all_stats()
                
                # 打印统计信息
                print(f"\n--- {datetime.now().strftime('%H:%M:%S')} ---")
                for stat in stats:
                    print(f"{stat['container']}: CPU={stat['cpu_percent']}%, "
                          f"MEM={stat['mem_percent']}% ({stat['mem_usage']}), "
                          f"PIDs={stat['pids']}")
                
                # 保存到CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for stat in stats:
                        writer.writerow([
                            stat['timestamp'], stat['container'], stat['cpu_percent'],
                            stat['mem_usage'], stat['mem_percent'], stat['net_io'],
                            stat['block_io'], stat['pids']
                        ])
                
                self.stats_history.extend(stats)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")
        
        self.monitoring = False
        print(f"\n监控数据已保存到: {csv_path}")
        return csv_path
    
    def generate_summary(self) -> Dict:
        """生成监控摘要"""
        if not self.stats_history:
            return {"error": "无监控数据"}
        
        summary = {}
        for container in self.container_names:
            container_stats = [s for s in self.stats_history if s['container'] == container]
            if not container_stats:
                continue
                
            cpu_values = [float(s['cpu_percent']) for s in container_stats]
            mem_values = [float(s['mem_percent']) for s in container_stats]
            
            summary[container] = {
                "samples": len(container_stats),
                "cpu": {
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "avg": sum(cpu_values) / len(cpu_values)
                },
                "memory": {
                    "min": min(mem_values),
                    "max": max(mem_values),
                    "avg": sum(mem_values) / len(mem_values)
                }
            }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Docker容器资源监控")
    parser.add_argument("--containers", nargs="+", 
                       default=["ragflow-ragflow-cpu-1", "ragflow-mysql-1", "ragflow-redis-1", "ragflow-es01-1"],
                       help="要监控的容器名称列表")
    parser.add_argument("--interval", type=int, default=5, help="采样间隔（秒）")
    parser.add_argument("--duration", type=int, default=600, help="监控时长（秒）")
    parser.add_argument("--output", 
                       default="/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单17/测试/性能分析",
                       help="输出目录")
    args = parser.parse_args()
    
    monitor = DockerResourceMonitor(args.containers, args.output)
    csv_path = monitor.monitor(args.interval, args.duration)
    
    # 生成摘要
    summary = monitor.generate_summary()
    summary_path = os.path.join(args.output, f"monitor_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n监控摘要已保存到: {summary_path}")
    print("\n=== 监控摘要 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
