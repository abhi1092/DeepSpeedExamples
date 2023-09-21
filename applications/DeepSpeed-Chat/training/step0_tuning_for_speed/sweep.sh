#!/bin/bash

# torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"\
# git pull && torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"\
git pull && deepspeed --autotuning tune --num_nodes=1 --num_gpus=8 \
 main.py\
    --model_name_or_path /new_data/rl-4-llm/granite_models/pretrained/step_225000_ckpt/\
  --data_path /new_data/datasets/summarization/tldr_sft_train_117k.jsonl \
  --prompt formatted_input\
  --chosen summary\
  --data_split 1,10,10\
  --data_output_path /app/.local_data/tuning \
  --max_seq_len 2048\
  --gradient_checkpointing\
  --seed 1234\
  --per_device_batch_size 1\
  --learning_rate 5e-5\
  --weight_decay 0.01\
  --deepspeed\
  --deepspeed_config ds_config_z3.json
