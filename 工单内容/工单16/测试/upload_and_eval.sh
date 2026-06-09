#!/bin/bash
# 上传文件到AutoDL服务器并运行评估

set -e

SERVER="root@connect.westc.seetacloud.com"
PORT=15717
PASSWORD="4hV1E7l+edMe"

# 本地路径
LOCAL_MODEL="/Users/suwente/.cache/modelscope/hub/models/Qwen/Qwen2-VL-2B-Instruct"
LOCAL_LORA="/Users/suwente/LLaMA-Factory/saves/qwen2-vl-2b/lora/sft"
LOCAL_DATA="/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单16/训练数据"
LOCAL_SCRIPT="/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单16/评估脚本/evaluate_vlm.py"

# 远程路径
REMOTE_DIR="/root/autodl-tmp/eval_qwen2vl"
REMOTE_MODEL="$REMOTE_DIR/model"
REMOTE_LORA="$REMOTE_DIR/lora"
REMOTE_DATA="$REMOTE_DIR/data"
REMOTE_SCRIPT="$REMOTE_DIR/evaluate_vlm.py"

echo "=== 1. 创建远程目录 ==="
expect << EOF
set timeout 30
spawn ssh -p $PORT -o StrictHostKeyChecking=no $SERVER
expect "password:"
send "$PASSWORD\r"
expect "\\$"
send "mkdir -p $REMOTE_DIR && echo 'Directory created'\r"
expect "\\$"
send "exit\r"
expect eof
EOF

echo -e "\n=== 2. 压缩模型文件 ==="
cd /Users/suwente/.cache/modelscope/hub/models/Qwen/
tar czf /tmp/qwen2vl-model.tar.gz Qwen2-VL-2B-Instruct/
echo "Model compressed: $(du -sh /tmp/qwen2vl-model.tar.gz | cut -f1)"

echo -e "\n=== 3. 压缩LoRA权重 ==="
cd /Users/suwente/LLaMA-Factory/saves/qwen2-vl-2b/lora/
tar czf /tmp/qwen2vl-lora.tar.gz sft/
echo "LoRA compressed: $(du -sh /tmp/qwen2vl-lora.tar.gz | cut -f1)"

echo -e "\n=== 4. 压缩评估数据 ==="
cd "/Users/suwente/Desktop/PDF智能问答RAG项目合集/工单16/"
tar czf /tmp/qwen2vl-data.tar.gz 训练数据/
echo "Data compressed: $(du -sh /tmp/qwen2vl-data.tar.gz | cut -f1)"

echo -e "\n=== 5. 上传文件到服务器 ==="
# 使用scp上传
expect << EOF
set timeout 300
spawn scp -P $PORT /tmp/qwen2vl-model.tar.gz $SERVER:$REMOTE_DIR/
expect "password:"
send "$PASSWORD\r"
expect "100%"
expect eof
EOF

expect << EOF
set timeout 60
spawn scp -P $PORT /tmp/qwen2vl-lora.tar.gz $SERVER:$REMOTE_DIR/
expect "password:"
send "$PASSWORD\r"
expect "100%"
expect eof
EOF

expect << EOF
set timeout 60
spawn scp -P $PORT /tmp/qwen2vl-data.tar.gz $SERVER:$REMOTE_DIR/
expect "password:"
send "$PASSWORD\r"
expect "100%"
expect eof
EOF

expect << EOF
set timeout 30
spawn scp -P $PORT "$LOCAL_SCRIPT" $SERVER:$REMOTE_SCRIPT
expect "password:"
send "$PASSWORD\r"
expect "100%"
expect eof
EOF

echo -e "\n=== 6. 解压文件并安装依赖 ==="
expect << EOF
set timeout 120
spawn ssh -p $PORT -o StrictHostKeyChecking=no $SERVER
expect "password:"
send "$PASSWORD\r"
expect "\\$"
send "cd $REMOTE_DIR && tar xzf qwen2vl-model.tar.gz && echo 'Model extracted'\r"
expect "\\$"
send "tar xzf qwen2vl-lora.tar.gz && echo 'LoRA extracted'\r"
expect "\\$"
send "tar xzf qwen2vl-data.tar.gz && echo 'Data extracted'\r"
expect "\\$"
send "pip install transformers peft pillow nltk rouge_score -q && echo 'Dependencies installed'\r"
expect "\\$"
send "exit\r"
expect eof
EOF

echo -e "\n=== 7. 运行评估 ==="
expect << EOF
set timeout 600
spawn ssh -p $PORT -o StrictHostKeyChecking=no $SERVER
expect "password:"
send "$PASSWORD\r"
expect "\\$"
send "cd $REMOTE_DIR && python evaluate_vlm.py 2>&1 | tee eval_output.log\r"
expect "\\$"
send "exit\r"
expect eof
EOF

echo -e "\n=== 8. 下载评估结果 ==="
expect << EOF
set timeout 60
spawn scp -P $PORT $REMOTE_DIR/eval_output.log /tmp/autodl_eval_output.log
expect "password:"
send "$PASSWORD\r"
expect "100%"
expect eof
EOF

echo -e "\n✅ 评估完成！结果已下载到 /tmp/autodl_eval_output.log"
