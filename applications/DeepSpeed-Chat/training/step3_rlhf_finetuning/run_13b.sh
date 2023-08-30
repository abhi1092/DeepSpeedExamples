#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

#run_13b.sh /new_data/delivery/granite13b/ckpt-800bn/sft600k/cft100k/checkpoint-3000/ /new_data/trained_models/opt-1.3-reward-dsc 3 3 /new_data/rlhf_out_8.17pm/

# DeepSpeed Team
ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

# Actor_Lr=5e-4
# Critic_Lr=5e-6
Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --master_port 12346 main.py \
   --print_answers \
   --enable_tensorboard \
   --offload_reference_model \
   --tensorboard_path $OUTPUT/tensorboard13b/ \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
<<<<<<< HEAD:applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/single_node/run_13b.sh
   --per_device_train_batch_size 4 \
   --per_device_mini_train_batch_size 4 \
   --gradient_accumulation_steps 8 \
   --generation_batch_numbers 1 \
=======
   --per_device_generation_batch_size 16 \
   --per_device_training_batch_size 16 \
   --generation_batches 1 \
>>>>>>> 32083e51c15a10fd2d90e62cd21e07e9219f8cf3:applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/opt/single_node/run_13b.sh
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --enable_hybrid_engine \
   --inference_tp_size 2 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --disable_actor_dropout \
   --rlhf_training \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log

    #    --actor_gradient_checkpointing \
#    --enable_ema \
