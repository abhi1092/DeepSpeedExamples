import argparse
import optuna
import subprocess

cmd = """torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--rdzv_id 101 \
--rdzv_endpoint 127.0.0.1:1111 \
main.py \
--print_loss \
--data_path /new_data/datasets/summarization-2/tldr_sft_train_117k.jsonl \
--data_split 10,0,0 \
--prompt formatted_input \
--chosen summary \
--model_name_or_path /new_data/rl-4-llm/granite_models/pretrained/step_225000_ckpt/ \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--max_seq_len 2048 \
--learning_rate 5e-6 \
--weight_decay 0.0 \
--num_train_epochs 1 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing \
--lr_scheduler_type cosine \
--num_warmup_steps 100 \
--seed 6742 \
--zero_stage 2 \
--deepspeed \
--optuna_study_name {study_name} \
--optuna_storage {database_url} \
--max_time 3600
"""

def parseargs():
  parser = argparse.ArgumentParser()
  parser.add_argument("--study_name", type=str, default="optuna_optimization")
  parser.add_argument("--n_trials", type=int, default=100)
  parser.add_argument("--database_url", type=str, required=True)
  parser.add_argument("--reset", action="store_true")
  return parser.parse_args()

def main():
  args = parseargs()
  if args.reset:
    optuna.delete_study(study_name=args.study_name, storage=args.database_url)
  
  study = optuna.create_study(
    study_name=args.study_name,
    storage=args.database_url,
    load_if_exists=True,
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3,
                                       n_warmup_steps=3
                                       ),
    )
  # check if the study did not exist before
  if len(study.trials) == 0:
    study.enqueue_trial({"learning_rate": 9e-6, "weight_decay": 1e-5})
    study.enqueue_trial({"learning_rate": 5e-6, "weight_decay": 1e-6})
  for _ in range(args.n_trials):
    formatted_cmd = cmd.format(study_name=args.study_name,
                               database_url=args.database_url)
    
    print(f'Running command:\n\n {formatted_cmd}\n\n =================== \n\n')
    # formatted_cmd = formatted_cmd.split()
    # ret_code = subprocess.Popen(formatted_cmd).wait()
    # if ret_code != 0:
    #   study.tell(trial, float("nan"))
    
if __name__ == "__main__":
  main()


