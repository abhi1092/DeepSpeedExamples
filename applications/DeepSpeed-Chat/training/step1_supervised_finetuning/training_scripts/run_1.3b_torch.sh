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

torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=8 --rdzv_id=102 --rdzv_endpoint="${MASTER_ADDR}:${NEW_PORT}" \
    main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-66b \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --output_dir $OUTPUT


