#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# 1B: 16 H100s for ≈ 2days
cd .. && CUR_DIR=$(pwd)

cd code && python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 25 \
--epochs 5 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir $CUR_DIR/zoo/HLLM-Books-neg512 \
--loss nce \
--MAX_TEXT_LENGTH 128 \
--dataset amazon_Video_Games \
--gradient_checkpointing True \
--text_keys '[\"title\",\"description\"]' \
--train_batch_size 4 \
--text_path $CUR_DIR/information \
--item_pretrain_dir $CUR_DIR/zoo/Qwen3-0.6B \
--user_pretrain_dir $CUR_DIR/zoo/Qwen3-0.6B \
--num_negatives 64