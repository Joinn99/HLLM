#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# 1B: 16 H100s for ≈ 2days

# for DOMAIN_NAME in Health_and_Household; do
# for SPLIT_NAME in pretrain phase1 phase2; do
# cd .. && CUR_DIR=/home/Data/
# cd HLLM
gpu_id=0
export master_port=$((29501+gpu_id))
export DOMAIN="Video_Games"
export SPLIT="merged"
export EVAL_NAME="merged_Vid_Spo-phase2-hllm-task"
cd code
CUDA_VISIBLE_DEVICES=${gpu_id} torchrun \
  --master_port=$master_port \
  --node_rank=0 \
  --nproc_per_node=1 \
  --nnodes=1 \
  run.py \
  --config_file \
  overall/LLM_ddp_full.yaml \
  HLLM/HLLM-test.yaml
