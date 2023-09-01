import base64
import json
import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import default_data_collator, AutoModelForCausalLM

import deepspeed
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from deepspeed.autotuning import Autotuner

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import load_hf_tokenizer, get_optimizer_grouped_parameters
from utils.model.model_utils import create_hf_model


def parse_args():
  parser = argparse.ArgumentParser(
    description = "autotune deepspeed for your model"
  )
  parser.add_argument('--model_name_or_path',
                      type=str,
  )
  
  parser.add_argument('--data_path',
                      type=str,
                      default=None,
                      help="Path for the dataset"
                      )
  
  parser.add_argument('--data_split',
                      type=str,
                      default="1,5,4", #firsh value should be small since this is just testing
                      help="Split of the dataset 1,1,1 means equal parts for all stages (this script uses only the first value)"
                    )
  parser.add_argument('--data_output_path',
                      type=str,
                      default="~/data_tmp",
                      help="here to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)"
  )
  parser.add_argument('--prompt_column_name',
                      type=str,
                      default="prompt",
                      help="Name of the column in the dataset that contains the prompt"
  )
  parser.add_argument('--max_seq_len',
                      type=int,
                      default=1024,
                      help="Maximum sequence length"
  )
  parser.add_argument('--seed',
                      type=int,
                      default=42,
                      help="Random seed"
  )
  parser.add_argument('--local_rank',
                      type=int,
                      default=-1,
                      help="Local rank for distributed training"
  )
  parser.add_argument('--disable_dropout',
                      action='store_true',
                      help="Disable dropout"
  )
  parser.add_argument('--per_device_batch_size',
                      type=int,
                      default=1,
                      help="Batch size per device")
  parser.add_argument('--learning_rate',
                      type=float,
                      default=5e-5,
                      help="Learning rate")
  parser.add_argument('--weight_decay',
                      type=float,
                      default=0.01,
                      help="Weight decay")
  parser.add_argument('--offload',
                      action='store_true',
                      help="Offload optimizer states to CPU"
  )
  parser = deepspeed.add_config_arguments(parser)
  args = parser.parse_args()

  return args
  
  
  

  
def main():
  args = parse_args()
  
  args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
  if args.local_rank == -1:
      device = torch.device("cuda")
  else:
      torch.cuda.set_device(args.local_rank)
      device = torch.device("cuda", args.local_rank)
      # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
      # torch.distributed.init_process_group(backend='nccl')
      deepspeed.init_distributed()
  
  args.global_rank = torch.distributed.get_rank()
  
  print(f'************{args}')

  tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                fast_tokenizer=True)
  
  ds_config = json.loads(base64.urlsafe_b64decode(args.deepspeed_config).decode('utf-8'))
  
  device = "cpu" if args.offload else "none"
  ds_config['zero_optimization']['offload_param']['device'] = device
  ds_config['zero_optimization']['offload_optimizer']['device'] = device
  
  model = create_hf_model(AutoModelForCausalLM,
                      args.model_name_or_path,
                      tokenizer,
                      ds_config,
                      disable_dropout=args.disable_dropout)

  # Split weights in two groups, one with weight decay and the other not.
  optimizer_grouped_parameters = get_optimizer_grouped_parameters(
      model, args.weight_decay)

  AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
  optimizer = AdamOptimizer(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            betas=(0.9, 0.95))

  model, _, _, _ = deepspeed.initialize(
      model=model,
      optimizer=optimizer,
      args=args,
      dist_init_required=True)
  
  if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    
  # Prepare dataset
  train_phase = 1
  dataset, _ = create_prompt_dataset(
    args.local_rank,
    args.data_path,
    args.data_split,
    args.data_output_path,
    train_phase,
    args.seed,
    tokenizer,
    args.max_seq_len,
    reload=True,
    prompt_column_name=args.prompt_column_name,
    )

  # DataLoaders creation:
  if args.local_rank == -1:
      sampler = SequentialSampler(dataset)
  else:
      sampler = DistributedSampler(dataset)
      
  dataloader = DataLoader(dataset,
                          collate_fn=default_data_collator,
                          sampler=sampler,
                          batch_size=args.per_device_batch_size)

  for step, batch in enumerate(dataloader):
      batch = to_device(batch, device)
      outputs = model(**batch, use_cache=False)
      loss = outputs.loss
      print(
          f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
      )
      model.backward(loss)
      model.step()
      
if __name__ == "__main__":
  main()