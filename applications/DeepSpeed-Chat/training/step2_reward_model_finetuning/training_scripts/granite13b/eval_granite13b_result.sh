# git pull && torchrun --nnodes=1 --node_rank=0 --nproc_per_node=5 --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
nohup git pull && torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:6543"\
 main.py --data_path /new_data/rlhf_samples/stage2_tldr/train.jsonl\
 --data_split 0,1,0\
 --model_name_or_path /new_data/rlhf_granite13b_stage2/\
 --per_device_train_batch_size 6\
 --per_device_eval_batch_size 6\
 --max_seq_len 1024\
 --learning_rate 9.65e-6\
 --weight_decay 0.1\
 --num_padding_at_beginning 0\
 --num_train_epochs 0\
 --gradient_accumulation_steps 1\
 --lr_scheduler_type cosine\
 --num_warmup_steps 100\
 --seed 1234\
 --zero_stage 3\
 --offload\
 --gradient_checkpointing\
 --rlhf_training\
 --deepspeed > /new_data/nohup_out_granite13b_rm_30-08-23.out 2>&1 &
#  --enable_tensorboard\
#  --tensorboard_path /new_data/rlhf_granite13b_stage2/tensorboard/\
#  --output_dir /new_data/rlhf_granite13b_stage2_erase_me/




 # --offload\
 #--gradient_checkpointing\

# deepspeed --autotuning run --num_nodes=1 --num_gpus=8 main.py --deepspeed /app/ds_config.json\
#  --data_path /new_data/rlhf_samples/stage2_tldr/train.jsonl\
#  --data_split 0,1,0\
#  --model_name_or_path /new_data/rl-4-llm/granite_models/pretrained/step_225000_ckpt/\
#  --per_device_train_batch_size 6\
#  --per_device_eval_batch_size 1\
#  --max_seq_len 1024\
#  --learning_rate 9.65e-6\
#  --weight_decay 0.1\
#  --num_padding_at_beginning 0\
#  --num_train_epochs 1\
#  --gradient_accumulation_steps 1\
#  --lr_scheduler_type cosine\
#  --num_warmup_steps 100\
#  --seed 1234\
#  --zero_stage 3\
#  --offload\
#  --gradient_checkpointing\
#  --enable_tensorboard\
#  --tensorboard_path /new_data/rlhf_granite13b_stage2/tensorboard/\
#  --output_dir /new_data/rlhf_granite13b_stage2/