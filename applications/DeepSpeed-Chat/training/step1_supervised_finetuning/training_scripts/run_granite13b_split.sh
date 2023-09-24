#!/bin/bash

function run_training() {
    split_index=$1
    cd /app/DeepSpeedExamples-internal/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
    git pull
    git checkout fixing_oom_step0
    HF_DATASETS_CACHE=/app/hf nohup torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    main.py \
   --print_loss \
   --data_path  /new_data/datasets/forca_092423_splits/train_${split_index}.jsonl \
   --data_output_path /new_data/deepspeed_cache_data/\
   --save_checkpoint \
   --load_checkpoint_path /new_data/granite_v2_forca_092123/deepspeed_checkpoint/\
   --model_name_or_path  /ai-models-cos/granite-13b-base-v1/step_300000_ckpt/\
   --data_split 1,0,0 \
   --prompt formatted_input\
   --chosen targets\
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 2048 \
   --learning_rate 5.47e-5 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 3 \
   --gradient_checkpointing \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 5739 \
   --zero_stage 2 \
   --deepspeed \
   --output_dir /new_data/granite_v2_forca_092123/ \
   --save_steps 100 > nohup_1.out 2>&1
}

for i in {0..4}
do
    run_training $i
    # Here, you might want to check the process status before proceeding to the next iteration.
    # For now, I'm assuming the command will finish successfully.
    sleep 10 # Add a small delay between iterations. Adjust as needed.
done
