#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

export MODEL_PATH=facebook/opt-350m
export ZERO_STAGE=3
export OUTPUT=/new_data/rlhf_out_8.17pm/
export Actor_Lr=9.65e-6
export Critic_Lr=5e-6

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --enable_tensorboard \
   --print_loss \
   --data_path  /new_data/datasets/summarization/tldr_sft_train_117k.jsonl \
   --data_split 10,0,0 \
   --model_name_or_path /new_data/rl-4-llm/granite_models/pretrained/step_225000_ckpt/ \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 8192 \
   --learning_rate 2.83e-5 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 6742 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --output_dir $OUTPUT 


