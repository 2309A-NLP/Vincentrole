#!/usr/bin/env python3
"""清空文档状态，强制重建向量索引"""
import shutil, os, json

base = "/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单13/output"

# 保留 graphml (实体/关系图谱) 和 LLM 缓存
keep = {"graph_chunk_entity_relation.graphml", "kv_store_llm_response_cache.json"}

for f in os.listdir(base):
    if f not in keep:
        path = os.path.join(base, f)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            os.remove(path)
            print(f"已删除: {f} ({size/1024:.0f}KB)")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"已删除目录: {f}")

# 确认剩余文件
print("\n剩余文件:")
for f in sorted(os.listdir(base)):
    sz = os.path.getsize(os.path.join(base, f)) if os.path.isfile(os.path.join(base, f)) else 0
    print(f"  {f} ({sz/1024:.0f}KB)")
