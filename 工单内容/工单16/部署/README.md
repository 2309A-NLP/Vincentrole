# 工单16 - 调试与微调专用视觉语言模型

## 任务目标

针对 RAGFlow 集成的 VLM 在工业专业问题上表现不佳的问题，完成 Qwen2-VL-2B-Instruct 模型微调的数据准备、训练调试和效果评估全流程。

## 项目结构

```
工单16/
├── 数据转换脚本/
│   └── convert_to_vlm_format.py   # IMDR QA → VLM JSONL 转换
├── 微调配置/
│   ├── qwen2vl_lora_sft.yaml      # LoRA 微调参数
│   ├── dataset_info.json          # 数据集注册
│   └── train.sh                   # 启动训练
├── 训练数据/
│   ├── patent_images/             # PDF 提取的图片
│   ├── vlm_train.jsonl            # 训练集 (80%)
│   └── vlm_eval.jsonl             # 评估集 (20%)
├── 评估脚本/
│   └── evaluate_vlm.py            # 专业评估（术语+图纸+BLEU/ROUGE）
├── 评估报告/
│   └── 评估报告.md                 # 评估结果文档
└── README.md
```

## 前置条件

- LLaMA-Factory（已安装）
- Qwen2-VL-2B-Instruct 模型（魔搭下载中）
- 通义千问 API Key（已配置）

## 使用流程

### 1. 数据准备
```bash
cd 数据转换脚本
python convert_to_vlm_format.py
```

### 2. 启动微调
```bash
cd 微调配置
bash train.sh
```

### 3. 评估
```bash
cd 评估脚本
python evaluate_vlm.py
```

## 评估维度

| 维度 | 说明 | 指标 |
|------|------|------|
| 专业术语准确性 | 工业术语（淬火、公差配合等）理解 | 选项匹配率 |
| 图纸推理正确性 | 专利图中部件位置/结构推理 | 选项匹配率 |
| BLEU/ROUGE | 文本生成质量 | N-gram 精确度 |
| 微调前后对比 | 基线 vs 微调后 | 全方位对比 |
