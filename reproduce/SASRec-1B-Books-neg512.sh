#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# batch_size = 16GPUs * 8 = 128
# flash attn need bf16
cd code && python3 main.py \
--config_file IDNet/llama_id.yaml overall/ID_deepspeed.yaml \
--optim_args.learning_rate 1e-3 \
--loss nce \
--train_batch_size 8 \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 201 \
--dataset amazon_Movies_and_TV \
--num_negatives 64 \
--item_embed_dim 512 \
--show_progress True \
--update_interval 100 \
--fix_temp True \
--optim_args.weight_decay 0.1 \
--user_pretrain_dir /home/Data/zoo/TinyLlama-1.1B-intermediate-step-1431k-3T \
--checkpoint_dir /home/Data/zoo/SASRec-1B-Books-neg512 \
--stopping_step 10 