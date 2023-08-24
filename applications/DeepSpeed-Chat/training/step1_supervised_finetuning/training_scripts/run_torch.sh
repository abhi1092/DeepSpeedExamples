#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

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
NEW_PORT=23457
#torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=2 --rdzv_id=102 --rdzv_endpoint="${MASTER_ADDR}:${NEW_PORT}" \
#    test.py


#torchrun --nnodes=1 --node_rank=${RANK} --nproc_per_node=8 --rdzv_id=102 --rdzv_endpoint="${MASTER_ADDR}:${NEW_PORT}" \
#    main.py \
#   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
#   --data_split 2,4,4 \
#   --model_name_or_path facebook/opt-66b \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --data_output_path /app/tmp \
#   --max_seq_len 512 \
#   --learning_rate 9.65e-6 \
#   --weight_decay 0. \
#   --num_train_epochs 16 \
#   --print_loss \
#   --gradient_accumulation_steps 1 \
#   --lr_scheduler_type cosine \
#   --num_warmup_steps 0 \
#   --seed 1234 \
#   --zero_stage $ZERO_STAGE \
#   --deepspeed \
#   --enable_tensorboard \
#   --tensorboard_path $OUTPUT \
#   --output_dir $OUTPUT

#deepspeed main.py \
#   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
#   --data_split 2,4,4 \
#   --model_name_or_path facebook/opt-66b \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --data_output_path /app/tmp \
#   --max_seq_len 512 \
#   --learning_rate 9.65e-6 \
#   --weight_decay 0. \
#   --num_train_epochs 16 \
#   --print_loss \
#   --gradient_accumulation_steps 1 \
#   --lr_scheduler_type cosine \
#   --num_warmup_steps 0 \
#   --seed 1234 \
#   --zero_stage $ZERO_STAGE \
#   --deepspeed \
#   --enable_tensorboard \
#   --tensorboard_path $OUTPUT \
#   --output_dir $OUTPUT
#/new_data/rl-4-llm/dpc_alignment/tulu_paper_reproduction/llama-7b-dolly
base_dir="/new_data/rl-4-llm/dpc_alignment"
experiment_dir="tulu_paper_reproduction/llama-7b-dolly-test"
granite_path="/new_data/rl-4-llm/experiment_alignment/granite13b_1000bn/cft_wadolly_100k_tulu_e2_beta_1e-6_base_700k_sft"
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_id=107 --rdzv_endpoint="${HOSTNAME}:${NEW_PORT}" \
    main.py \
   --data_path dolly_dataset \
   --data_split 1,0,0,0 \
   --model_name_or_path  decapoda-research/llama-7b-hf \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --data_output_path ./data \
   --max_seq_len 2048 \
   --learning_rate 2e-5 \
   --weight_decay 0.0 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type linear \
   --warmup_percentage 0.03 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 2 \
   --offload \
   --deepspeed \
   --output_dir $base_dir/$experiment_dir