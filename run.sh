#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# 1B: 16 H100s for ≈ 2days

DOMAIN="Movies_and_TV"
PHASE="pretrain"

cd .. && CUR_DIR=/home/Data/
cd HLLM
cd code && python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM-${PHASE}.yaml \
--MAX_ITEM_LIST_LENGTH 25 \
--epochs 3 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir $CUR_DIR/zoo/HLLM-${DOMAIN}-${PHASE} \
--loss nce \
--MAX_TEXT_LENGTH 64 \
--dataset amazon_${DOMAIN} \
--gradient_checkpointing True \
--text_keys '[\"title\",\"description\"]' \
--train_batch_size 16 \
--text_path $CUR_DIR/HLLM/information \
--item_pretrain_dir $CUR_DIR/zoo/Qwen3-0.6B \
--user_pretrain_dir $CUR_DIR/zoo/Qwen3-0.6B \
--num_negatives 512