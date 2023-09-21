#!/bin/bash
  main.py --data_path /new_data/datasets/flan/erase.jsonl \
  --model_name_or_path OpenAssistant/falcon-7b-sft-top1-696 \
# git pull && torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"\
nohup git pull && torchrun --nnodes=1 --node_rank=0 --nproc_per_node=5 --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  main.py   --data_path /new_data/datasets/flan/t0_30k.jsonl \
  --model_name_or_path OpenAssistant/falcon-40b-sft-top1-560 \
  --prompt_column_name formatted_input \
  --data_split 0,0,1 \
  --data_output_path /app/.local_data/shiv_data \
  --per_device_batch_size 2 \
  --max_prompt_seq_len 1600 \
  --max_seq_len 2048 \
  --temperature 0.7 \
  --top_k 50 \
  --top_p .9 \
  --repetition_penalty 1.2 \
  --num_answers_per_prompt 1 \
  --output_dir /new_data/rlhf_samples/oasst_flan_t0_30k_filtered \
  --seed 567290 \
  --zero_stage 3 \
  --offload \
  --disable_dropout \
  --deepspeed > /new_data/nohup_out_sample_falcon.out 2>&1 &

  #nohup ping google and redirect both stdout and sterr
  #nohup ping google.com > /dev/null 2>&1 &