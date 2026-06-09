#!/bin/bash
# 工单16 - 启动 LoRA 微调
# 用法: bash train.sh

set -e

# 切换到 LLaMA-Factory 目录
cd /Users/suwente/LLaMA-Factory

# 激活环境（用 llama-factory-311）
PYTHON=/Users/suwente/anaconda3/envs/llama-factory-311/bin/python

# 训练数据路径
DATA_DIR=/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单16/训练数据
CONFIG_DIR=/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单16/微调配置

# 软链接数据集（LLaMA-Factory 需要 data/ 目录下）
ln -sf "$DATA_DIR/vlm_train.jsonl" data/vlm_train.jsonl 2>/dev/null || true
ln -sf "$DATA_DIR/vlm_eval.jsonl" data/vlm_eval.jsonl 2>/dev/null || true

# 用自定义 dataset_info.json（合并系统默认的）
$PYTHON -c "
import json
import os
base = json.load(open('data/dataset_info.json'))
custom = json.load(open('$CONFIG_DIR/dataset_info.json'))
base.update(custom)
json.dump(base, open('data/dataset_info.json', 'w'), ensure_ascii=False, indent=2)
print('dataset_info.json 已合并')
"

echo "=========================================="
echo "启动 LoRA 微调 - Qwen2-VL-2B-Instruct"
echo "训练数据: $DATA_DIR/vlm_train.jsonl"
echo "配置: $CONFIG_DIR/qwen2vl_lora_sft.yaml"
echo "=========================================="

# 启动训练
$PYTHON -m llamafactory.cli.train "$CONFIG_DIR/qwen2vl_lora_sft.yaml"

echo "✅ 训练完成！"
echo "模型保存在: /Users/suwente/LLaMA-Factory/saves/qwen2-vl-2b/lora/sft"
