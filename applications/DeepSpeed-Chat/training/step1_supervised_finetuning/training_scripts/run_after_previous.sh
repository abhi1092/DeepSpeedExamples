#!/bin/bash

# Define the command to be executed
cmd() {
    CMD="sleep 20 &&\
    cd /app/DeepSpeedExamples-internal/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/ &&\
    git pull &&\
    HF_DATASETS_CACHE=/app/hf torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint=\"${MASTER_ADDR}:${MASTER_PORT}\" \
        main.py \
        --print_loss \
        --data_path  /new_data/datasets/forca/trainmix/forca_train_mix_145k.jsonl \
        --max_num_per_split 180000\
        --data_output_path /new_data/deepspeed_cache_data/\
        --save_checkpoint \
        --model_name_or_path  /ai-models-cos/granite-13b-base-v1/step_300000_ckpt/\
        --data_split 1,0,0 \
        --prompt formatted_input\
        --chosen targets\
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --max_seq_len 2048 \
        --learning_rate 6.928e-5 \
        --weight_decay 0. \
        --num_train_epochs 3 \
        --gradient_accumulation_steps 3 \
        --gradient_checkpointing \
        --lr_scheduler_type cosine \
        --num_warmup_steps 20 \
        --seed 5739 \
        --zero_stage 2 \
        --deepspeed \
        --output_dir /new_data/granite_v1_forcav1_good_warmup/ \
        --save_steps 100"
    echo -e $CMD
    eval $CMD
}

# Wait for all torchrun processes to finish
while pgrep -x "torchrun" > /dev/null
# while ps aux | grep '[s]leep 300' > /dev/null
do
    echo "Waiting for existing torchrun processes to finish..."
    sleep 60
done

# Execute the command
cmd