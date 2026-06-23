# Linly-Talker + 医疗RAG

**数字人交互 + 医疗知识库检索增强生成系统**

基于 Linly-Talker 框架，集成医疗问答知识库，实现带数字人形象的医疗咨询对话系统。

## 项目结构

```
Linly-Talker-医疗RAG/
├── 设计/          # 架构文档、UI截图、设计说明
├── 研发/          # 核心代码（.py、模块源码）
├── 测试/          # 医疗问答测试数据、示例输入
├── 部署/          # 启动脚本、安装依赖、环境配置
├── 优化/          # 优化方案、补丁说明
└── README.md      # 本文件
```

## 快速启动

```bash
# 1. 安装依赖
cd 部署 && bash download_models.sh

# 2. 启动服务
cd .. && bash 部署/start_rag.sh
```

## 医疗 RAG 核心

- **向量模型**: BGE-small-zh-v1.5（本地嵌入检索）
- **知识库**: 医疗问答数据集（kb_score5.jsonl，约 15MB）
- **构建脚本**: build_kb.py（将 QA 数据向量化）
- **集成入口**: webui.py（LLM/QwenRAG.py 中实现检索增强）

## 技术栈

- **数字人**: SadTalker / MuseTalk / Wav2Lip
- **语音**: CosyVoice / GPT-SoVITS / Edge-TTS / FunASR
- **语言模型**: Qwen / ChatGLM / GPT 等
- **向量检索**: BGE-small-zh + NumPy

> 注意：大模型文件（checkpoints 等约 20GB）未包含在此包中，请运行 `部署/download_models.sh` 自动下载。
