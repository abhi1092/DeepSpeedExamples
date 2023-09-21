#!/bin/bash
# git pull && torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"\
git pull && torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
 main.py   --data_path /new_data/datasets/summarization/tldr_sft_train_117k.jsonl \
  --model_name_or_path /new_data/dpc-st1-summarization-4/step_2313 \
  --data_split 0,0,1 \
  --data_output_path /app/.local_data/tldr_data \
  --per_device_batch_size 24 \
  --max_prompt_seq_len 512 \
  --max_seq_len 1024 \
  --temperature 0.7 \
  --top_k 50 \
  --top_p .9 \
  --repetition_penalty 1.2 \
  --num_return_sequences 2 \
  --output_dir /new_data/rlhf_samples/step_2313 \
  --seed 567290 \
  --zero_stage 2 \
  --disable_dropout \
  --deepspeed

  #      Disable dropout \
  #   --offload             Offload model to CPU

  from deepspeed import DeepSpeedInferenceConfig