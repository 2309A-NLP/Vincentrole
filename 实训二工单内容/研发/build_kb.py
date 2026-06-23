# -*- coding: utf-8 -*-
"""用 bge-small-zh 在本地GPU把医疗问答库向量化，存成 npy + json。"""
import json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

BGE_DIR  = "/root/.cache/modelscope/hub/AI-ModelScope/bge-small-zh-v1___5"
KB       = "/root/Linly-Talker/kb_score5.jsonl"
OUT_EMB  = "/root/Linly-Talker/kb_emb.npy"
OUT_META = "/root/Linly-Talker/kb_meta.json"
BATCH    = 256
MAXLEN   = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

tok = AutoTokenizer.from_pretrained(BGE_DIR)
model = AutoModel.from_pretrained(BGE_DIR).to(device).eval().half()

meta, texts = [], []
for line in open(KB, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    meta.append(d)
    texts.append(d["q"])          # 检索用问题字段
print("待向量化条数:", len(texts))

embs = []
t0 = time.time()
for i in range(0, len(texts), BATCH):
    batch = texts[i:i + BATCH]
    enc = tok(batch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
        e = out.last_hidden_state[:, 0]                       # bge 用 CLS pooling
        e = torch.nn.functional.normalize(e.float(), p=2, dim=1)
    embs.append(e.cpu().numpy())
    if (i // BATCH) % 10 == 0:
        print(f"  {i}/{len(texts)}  {time.time()-t0:.0f}s", flush=True)

emb = np.concatenate(embs, axis=0).astype("float32")
np.save(OUT_EMB, emb)
json.dump(meta, open(OUT_META, "w", encoding="utf-8"), ensure_ascii=False)
print("DONE shape=", emb.shape, "用时", round(time.time() - t0, 1), "s")
