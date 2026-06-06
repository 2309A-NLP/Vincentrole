#!/usr/bin/env python3
"""测试bge模型离线加载 - 使用本地路径"""
import os
import sys
sys.path.insert(0, "/Users/suwente/.hermes/hermes-agent/venv/lib/python3.13/site-packages")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 直接用快照路径
model_path = "/Users/suwente/.cache/huggingface/hub/models--BAAI--bge-base-zh-v1.5/snapshots/f03589ceff5aac7111bd60cfc7d497ca17ecac65"

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(model_path, device="cpu")
print("DIM:", m.get_sentence_embedding_dimension())
emb = m.encode(["测试文本"], normalize_embeddings=True, show_progress_bar=False)
print("OK, shape:", emb.shape)
