# deepdoc 模块技术分析文档

## 一、概述

deepdoc 是 RAGFlow 的核心文档解析引擎，负责将各类格式的文档（PDF、Word、Excel、图片等）进行深度解析、分块、向量化并建立索引。deepdoc 采用**布局分析（Layout Analysis）+ OCR + 表格识别**的多模态解析策略，尤其针对低质量、图片型PDF提供深度解析能力。

---

## 二、DeepDoc 内置解析器清单

| 解析器 | 文件 | 支持的文件类型 |
|--------|------|---------------|
| PDF解析器 | `deepdoc/parser/pdf_parser.py` | PDF（文字型+图片型） |
| Docx解析器 | `deepdoc/parser/docx_parser.py` | .docx |
| Excel解析器 | `deepdoc/parser/excel_parser.py` | .xlsx |
| PPT解析器 | `deepdoc/parser/ppt_parser.py` | .pptx |
| 图片解析器 | `deepdoc/parser/figure_parser.py` | .jpg, .png, .bmp |
| HTML解析器 | `deepdoc/parser/html_parser.py` | .html |
| Markdown解析器 | `deepdoc/parser/markdown_parser.py` | .md |
| TXT解析器 | `deepdoc/parser/txt_parser.py` | .txt |
| JSON解析器 | `deepdoc/parser/json_parser.py` | .json |
| EPUB解析器 | `deepdoc/parser/epub_parser.py` | .epub |
| PaddleOCR解析器 | `deepdoc/parser/paddleocr_parser.py` | 通用OCR |
| MinerU解析器 | `deepdoc/parser/mineru_parser.py` | MinerU增强解析 |
| Docling解析器 | `deepdoc/parser/docling_parser.py` | Docling格式 |

---

## 三、PDF 解析策略（四种分块策略）

RAGFlow 通过 `rag/app/` 目录下的不同应用模块实现不同的解析策略，根据 `paper_id` 字段选择：

### 1. paper 模式（学术论文）

**文件：** `rag/app/paper.py`

- **适用场景：** 学术论文、技术文档
- **解析流程：**
  1. 使用 `RAGFlowPdfParser` 进行布局分析
  2. 识别标题、作者、摘要、正文、参考文献等结构
  3. 按章节层级分块，保留语义连续性
  4. 每块附带元数据（页码、章节名、层级）
- **分块策略：** 按章节自然段落切分，大段落按 token 数二次分割

### 2. table 模式（表格密集文档）

**文件：** `rag/app/table.py`

- **适用场景：** 财务报表、数据报表、说明书
- **解析流程：**
  1. 使用 `TableStructureRecognizer` 识别表格区域
  2. 对表格区域进行行列解析
  3. 非表格区域按文本流处理
  4. 表格保持原格式，每行/每格独立编址
- **分块策略：** 表格为一个独立 chunk，非表格按段落切分

### 3. one 模式（单一文档）

**文件：** `rag/app/one.py`

- **适用场景：** 统一格式的整篇文档
- **解析流程：**
  1. 整体文档一次性解析
  2. 不做复杂结构分析
  3. 按固定 token 数简单切分（默认 512 tokens）
- **分块策略：** 固定大小滑动窗口，带重叠（overlap=64 tokens）

### 4. knowledge_graph 模式（知识图谱）

**文件：** `rag/app/naive.py`（结合 graphrag 模块）

- **适用场景：** 需要构建知识图谱的文档
- **解析流程：**
  1. 先进行基础文本解析
  2. 提取实体关系
  3. 构建知识图谱索引
- **分块策略：** 在基础分块基础上，额外建立实体-关系索引

### 策略选择机制

解析策略选择在 `rag/app/__init__.py` 中实现，通过映射表 `PARSER_MAP` 将 `parser_id` 映射到对应的处理模块：

```
PARSER_MAP = {
    "paper": paper,
    "table": table,
    "one": one,
    "knowledge_graph": naive,
    ...
}
```

---

## 四、任务触发机制（Redis Stream）

### 触发流程

```
用户点击"解析"按钮
        ↓
RAGFlow API Server 创建 Document 记录
        ↓
将解析任务推送至 Redis Stream（key: "tasks"）
        ↓
Task Executor 从 Redis Stream 消费任务
        ↓
调用 do_handle_task 处理
```

### Redis Stream 消息格式

```json
{
    "task_id": "uuid",
    "doc_id": "document_uuid",
    "parser_id": "paper|table|one|knowledge_graph",
    "tenant_id": "tenant_uuid",
    "chunk_count": 0,
    "progress": 0.0,
    "status": "RUNNING"
}
```

### Task Service（数据库层）

通过 `api/db/services/task_service.py` 的 `TaskService` 管理任务持久化：
- `create_task()`: 创建解析任务
- `update_progress()`: 更新进度
- `has_canceled()`: 检查是否被取消
- `get_tasks()`: 获取任务列表

---

## 五、do_handle_task 主要逻辑（1944 行）

**文件：** `rag/svr/task_executor.py`

### 核心流程

```
do_handle_task(task_info)
    │
    ├─ 1. 获取 Document 元数据（格式、大小、当前解析状态）
    │
    ├─ 2. 根据 parser_id 选择解析器
    │      paper → rag/app/paper.py
    │      table → rag/app/table.py
    │      one   → rag/app/one.py
    │      knowledge_graph → rag/app/naive.py
    │      ...
    │
    ├─ 3. 调用解析器的 __call__ 方法
    │      ├─ PDF → deepdoc/parser/pdf_parser.py (OCR + 布局分析)
    │      ├─ Docx → python-docx 提取
    │      ├─ Excel → openpyxl 提取
    │      └─ 图片 → PaddleOCR 识别
    │
    ├─ 4. 文本分块（chunking）
    │      ├─ 按策略（paper/table/one/naive）
    │      ├─ 设置 chunk_size（默认 512）
    │      └─ 设置 overlap（默认 64）
    │
    ├─ 5. 向量化（Embedding）
    │      ├─ 调用 Embedding 模型（BGE/通义千问等）
    │      └─ 生成向量存储到向量库（ES/Infinity）
    │
    ├─ 6. 更新任务进度
    │      ├─ 写入 TaskService
    │      └─ 更新 Document 状态为完成
    │
    └─ 7. 触发后处理
           ├─ Raptor（如配置）
           ├─ 知识图谱（如配置）
           └─ 元数据提取
```

### 关键技术实现

| 阶段 | 技术 | 说明 |
|------|------|------|
| 布局分析 | ONNX Runtime | 使用训练好的布局识别模型（XGBoost）检测文本块、表格、图片 |
| OCR | PaddleOCR | 识别图片中的文字，支持中英文 |
| 表格识别 | TableStructureRecognizer | 基于深度学习的表格结构识别 |
| 分块 | 滑动窗口 | 按 token 数切分 + 语义保留 |
| 向量化 | TEI / API Embedding | 支持 BGE、通义千问等 |
| 任务队列 | Redis Stream | 支持多消费者、消息持久化、ACK机制 |
| 并发控制 | Semaphore + ThreadPool | 支持多 GPU 并行处理 |

---

## 六、PDF 解析核心技术

### 1. 布局分析（Layout Analysis）

`RAGFlowPdfParser` 使用 **ONNX Runtime** 加载预训练的布局识别模型，识别以下区域类型：
- **Text**：正文文本块
- **Title**：标题
- **Table**：表格区域
- **Figure**：图片区域
- **Footer/Header**：页眉页脚
- **Reference**：参考文献区域

### 2. OCR 文字识别

对于图片型PDF（如本工单中的 `CN100342976C.pdf`，8页全为图片），deepdoc 调用 **PaddleOCR** 进行文字提取：
- 支持中英文混合识别
- 保留文字在原图中的坐标位置
- 坐标用于后续的布局还原和段落合并

### 3. 图片-文字关联

识别出图片区域后，deepdoc 会：
1. 提取图片中的文字（OCR）
2. 记录图片在页面中的位置坐标
3. 为图片生成文字描述（通过 VLM）
4. 文字描述与 OCR 结果合并作为该区域的文本内容

### 4. 针对低质量PDF的优化

- **PDF 还原度检测**：通过文本提取率和布局分析判断 PDF 是否为图片型
- **自动降级策略**：文字型 PDF 用 pdfplumber 提取，失败则自动降级为 OCR 模式
- **多分辨率处理**：低分辨率 PDF 自动放大后处理

---

## 七、代码位置

| 组件 | 路径 | 行数 |
|------|------|------|
| PDF 解析核心 | `deepdoc/parser/pdf_parser.py` | 2079 行 |
| 任务执行器 | `rag/svr/task_executor.py` | 1944 行 |
| 布局识别 | `deepdoc/vision/` | - |
| OCR | `deepdoc/vision/OCR.py` | - |
| Paper 解析 | `rag/app/paper.py` | - |
| Table 解析 | `rag/app/table.py` | - |
| One 解析 | `rag/app/one.py` | - |
| Naive 解析 | `rag/app/naive.py` | - |
| 任务服务 | `api/db/services/task_service.py` | - |
| 应用路由 | `rag/app/__init__.py` | - |
