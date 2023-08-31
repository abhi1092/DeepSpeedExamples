#!/bin/bash
CLM_PATH=/app/rl-llm-support-libs/transformers/examples/pytorch/language-modeling/run_clm.py
MODEL_PATH=/new_data/rl-4-llm/granite_models/pretrained/step_225000_ckpt/
DS_CONFIG_PATH=ds_config_z3.json

deepspeed --autotuning tune --num_nodes=1 --num_gpus=8 $CLMPATH \
  --deepspeed $DS_CONFIG_PATH \
  --model_name_or_path $MODEL_PATH \
  --do_train \
  --do_eval \
  --fp16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1