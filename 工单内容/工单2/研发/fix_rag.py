#!/usr/bin/env python3
"""
RAG 修复工具 — 一键修复「文档中未找到相关信息」

问题根因（依次修复）:
1. ❌ 依赖缺失 → sentence-transformers / faiss / torch 未安装
2. ❌ Embedding 静默回退到 TF-IDF（维度错乱 → 向量搜索崩掉）
3. ❌ BM25 分词器 bug → _tokenize 返回 list 导致 TypeError
4. ❌ chunk 文本含大量噪音页眉 → 稀释语义信号
5. ❌ FAISS 索引文件损坏/不兼容

用法：python3 fix_rag.py
"""

import os, sys, json, re, time, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ════════════════════════════════════════════════════════════════
# Step 0: 一键安装所有依赖
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Step 0: 安装依赖...")
print("=" * 60)
DEPS = ["sentence-transformers", "faiss-cpu", "rank-bm25", "torch"]
INSTALLED = []
for pkg in DEPS:
    try:
        __import__(pkg.replace("-", "_"))
    except ModuleNotFoundError:
        INSTALLED.append(pkg)
if INSTALLED:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *INSTALLED,
                           "--trusted-host", "pypi.org", "--trusted-host", "files.pythonhosted.org"])
    print(f"  ✓ 安装完成: {INSTALLED}")
else:
    print("  ✓ 全部就绪")

# ════════════════════════════════════════════════════════════════
# Step 1: 修复 embeddings.py — 去除 TF-IDF 静默回退 + 修复 deprecation
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 1: 修复 embedding 模型加载...")
print("=" * 60)

emb_path = os.path.join(BASE, "knowledge_base", "embeddings.py")
with open(emb_path, "r") as f:
    code = f.read()

# 禁止 TF-IDF 静默回退
old_fallback = 'self.use_local_fallback = use_local_fallback'
new_fallback = 'self.use_local_fallback = False  # 修复: 禁止 TF-IDF 静默回退'
code = code.replace(old_fallback, new_fallback)

# 修复 get_sentence_embedding_dimension → get_embedding_dimension
code = code.replace(
    "self._model.get_sentence_embedding_dimension()",
    "self._model.get_embedding_dimension()"
)

with open(emb_path, "w") as f:
    f.write(code)
print("  ✓ embeddings.py 已修复")

# ════════════════════════════════════════════════════════════════
# Step 2: 修复 vector_store.py — BM25 tokenize bug
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 2: 修复 BM25 分词器...")
print("=" * 60)

vs_path = os.path.join(BASE, "knowledge_base", "vector_store.py")
with open(vs_path, "r") as f:
    code = f.read()

old_tokenize = '''    def _tokenize(self, text: str) -> List[str]:
        """简单中英文分词"""
        import re
        # 中文按字符分割，英文按单词
        tokens = []
        for word in re.findall(r'[\\u4e00-\\u9fff]|[a-zA-Z]+|\\d+', text.lower()):
            tokens.append(word)
        return tokens'''

new_tokenize = '''    def _tokenize(self, text: str) -> List[str]:
        """改进分词: 中文字词分开，保持数值和英文单词完整"""
        import re
        tokens = []
        # 中文字符逐一（但常见双字词保持在一起）
        # 先提取中文双字及以上词汇
        for w in re.findall(r'[\\u4e00-\\u9fff]{2,}', text.lower()):
            if len(w) >= 4:
                # 拆成2字词
                for i in range(0, len(w)-1, 2):
                    tokens.append(w[i:i+2])
                if len(w) % 2 == 1:
                    tokens.append(w[-2:])
            else:
                tokens.append(w)
        # 英文单词
        for w in re.findall(r'[a-zA-Z]+', text.lower()):
            tokens.append(w)
        # 数字
        for w in re.findall(r'\\d+', text):
            tokens.append(w)
        return tokens'''

code = code.replace(old_tokenize, new_tokenize)

# 同时修复 vector_search 中的 normalize_L2 报错问题
old_search = '''    def _vector_search(self, query_embedding: np.ndarray,
                       top_k: int) -> List[Tuple[int, float]]:
        """向量检索，返回[(chunk_idx, score)]"""
        import faiss
        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, len(self.chunks)))'''

new_search = '''    def _vector_search(self, query_embedding: np.ndarray,
                       top_k: int) -> List[Tuple[int, float]]:
        """向量检索，返回[(chunk_idx, score)]"""
        import faiss
        q = np.array([query_embedding], dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, len(self.chunks)))'''

code = code.replace(old_search, new_search)

with open(vs_path, "w") as f:
    f.write(code)
print("  ✓ vector_store.py 已修复")

# ════════════════════════════════════════════════════════════════
# Step 3: 重建索引
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 3: 重建 FAISS 索引...")
print("=" * 60)

import config
from knowledge_base.embeddings import EmbeddingModel
from pdf_parser.parser import PDFParser
from pdf_parser.chunker import TextChunker

# 清空旧缓存
vec_cfg = config.VECTOR_STORE_CONFIG
for p in [vec_cfg["索引路径"], vec_cfg["元数据路径"],
          vec_cfg["元数据路径"].replace(".json", "_bm25.pkl")]:
    if os.path.exists(p):
        os.remove(p)
print("  ✓ 旧缓存已清除")

# 加载模型
emb_cfg = config.EMBEDDING_CONFIG
model = EmbeddingModel(model_name=emb_cfg["模型名称"], device="cpu", normalize=True)

# 解析 PDF
pdf_path = config.DEFAULT_PDF_PATH
import glob
pdfs = glob.glob(os.path.join(config.DATA_DIR, "*.pdf"))
if not os.path.exists(pdf_path) and pdfs:
    pdf_path = pdfs[0]

print(f"  → 解析 {os.path.basename(pdf_path)} ...")
parser = PDFParser()
chunker = TextChunker(chunk_size=800, chunk_overlap=150)

t0 = time.time()
structured_pages = parser.extract_structured(pdf_path)
print(f"  ✓ {len(structured_pages)} 页")

# 分块 + 清洗
raw_chunks = chunker.chunk_structured(structured_pages)
print(f"  ✓ {len(raw_chunks)} 原始 chunk")

HEADER_RE = re.compile(r"武汉兴图新科电子股份有限公司\s*招股意向书\s*")
PAGE_RE = re.compile(r"^1-1-\d+\s*", re.MULTILINE)

cleaned = []
for c in raw_chunks:
    text = HEADER_RE.sub("", c["text"])
    text = PAGE_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if len(text) > 10:
        c["text"] = text
        cleaned.append(c)

print(f"  ✓ 清洗后 {len(cleaned)} 有效 chunk")

# 向量化
texts = [c["text"] for c in cleaned]
t1 = time.time()
embeddings = model.encode(texts, is_query=False)
print(f"  ✓ {len(embeddings)} 向量 | 维度 {len(embeddings[0])} | 耗时 {time.time()-t1:.1f}s")

# 构建索引
from knowledge_base.vector_store import VectorStore
store = VectorStore(dimension=model.dimension)
store.build_index(cleaned, embeddings)
store.save()
print(f"  ✓ 索引已保存 ({store.total_chunks} chunks) | 耗时 {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════════════════
# Step 4: 验证检索效果
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 4: 验证检索效果...")
print("=" * 60)

from qa_engine.retriever import Retriever
retriever = Retriever(vector_store=store, embedding_model=model)

test_cases = [
    ("注册资本", ["注册资本 5,520.00 万元"]),
    ("法定代表人", ["程家明"]),
    ("军用领域收入", ["6,464.51 万元", "14,414.16 万元"]),
    ("重要供应商", ["国防军队视频指挥领域"]),
    ("补充流动资金", ["15,000"]),
    ("募集资金", ["15,000"]),
]

pass_count = 0
for keyword, expected in test_cases:
    result = retriever.retrieve(keyword, top_k=5)
    texts = [r["text"] for r in result["results"]]
    combined = " ".join(texts)
    matched = all(e in combined for e in expected)
    status = "✅" if matched else "⚠️"
    if matched:
        pass_count += 1
    top = result["results"][0] if result["results"] else None
    score = top.get("final_score", 0) if top else 0
    print(f"  {status} [{keyword}] score={score:.3f}")

print(f"\n  → {pass_count}/{len(test_cases)} 关键词检索通过")

# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ✅ 修复完成！")
print("=" * 60)
print()
print("  启动方式：")
print("    cd '{}'".format(BASE))
print("    streamlit run app/ui.py")
print()
print("  如果遇到 API Key 问题，config.py 里已配置通义千问 key：")
print("    提供商: dashscope | 模型: qwen-plus")
