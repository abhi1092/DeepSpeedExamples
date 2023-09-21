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

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import load_hf_tokenizer, get_optimizer_grouped_parameters, print_rank_0, get_column_names, set_random_seed, to_device
from utils.model.model_utils import create_hf_model
from utils.data.data_utils import create_prompt_dataset


def parse_args():
  parser = argparse.ArgumentParser(
    description = "autotune deepspeed for your model"
  )
  parser.add_argument('--model_name_or_path',
                      type=str,
  )
  
  parser.add_argument('--data_path',
                      nargs='*',
                      default=[None],
                      help="list of datasets, if many they are concatenated"
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
  parser.add_argument('--prompt',
                      type=str,
                      default=None,
                      help="Name of the column in the dataset that contains the prompt"
  )
  parser.add_argument('--chosen',
                      type=str,
                      default=None,
                      help="Name of the column in the dataset that contains the chosen answer"
  )
  parser.add_argument('--rejected',
                      type=str,
                      default=None,
                      help="Name of the column in the dataset that contains the rejected answer"
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
  
  parser.add_argument("--gradient_checkpointing",
                      action="store_true",
                      help="Enable gradient checkpointing")
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
  
def set_deepspeed_config(args, ds_config):
  if args.offload:
    if 'zero_optimization' not in ds_config:
      ds_config['zero_optimization'] = {}
    if 'offload_param' not in ds_config['zero_optimization']:
      ds_config['zero_optimization']['offload_param']= {'device': 'cpu',
                                                        'pin_memory': True}
      ds_config['zero_optimization']['offload_optimizer']= {'device': 'cpu',
                                                            'pin_memory': True}
  ds_config['train_micro_batch_size_per_gpu'] = args.per_device_batch_size
  return ds_config
  
  
def main():
  args = parse_args()
  
  args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
  args.column_names = get_column_names(args)
  if args.local_rank == -1:
      device = torch.device("cuda")
  else:
      torch.cuda.set_device(args.local_rank)
      device = torch.device("cuda", args.local_rank)
      # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
      # torch.distributed.init_process_group(backend='nccl')
      deepspeed.init_distributed()
  
  args.global_rank = torch.distributed.get_rank()
  print_rank_0(f'column_names: {args.column_names}', color="GREEN")
  
  print(f'************{args}')
  set_random_seed(args.seed)
  
  torch.distributed.barrier()

  tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                fast_tokenizer=True)
  
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
    column_names=args.column_names,
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
  
  # if Path(args.deepspeed_config).is_file():
  #   with open(args.deepspeed_config, 'r') as f:
  #     ds_config = json.load(f)
  # elif args.deepspeed_config is not None:
  #   ds_config = json.loads(base64.urlsafe_b64decode(args.deepspeed_config).decode('utf-8'))
  if args.deepspeed_config is not None:
    try:
      # Try to decode the value as base64
      ds_config = json.loads(base64.urlsafe_b64decode(args.deepspeed_config).decode('utf-8'))
    except:
      print_rank_0("Failed to decode deepspeed config as base64, assuming it's a file path", color="YELLOW")
      with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
  
  ds_config = set_deepspeed_config(args, ds_config)
  from pprint import pprint
  if args.local_rank == 0 or args.local_rank == -1:
    print_rank_0("DeepSpeed Config:", color="GREEN")
    pprint(ds_config)
    pprint(args)
    
  args.deepspeed_config = ds_config
  
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

  model, optimizer, _, _ = deepspeed.initialize( #should I add the optimizer here?
      model=model,
      optimizer=optimizer,
      args=args,
      dist_init_required=True)
  
  if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

  model.train()
  print_rank_0(f'number of batches {len(dataloader)}', color="GREEN")
  for step, batch in enumerate(dataloader):
      batch = to_device(batch, device)
      outputs = model(**batch, use_cache=False)
      loss = outputs.loss
      print(
          f"Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
      )
      model.backward(loss)
      model.step()
      
if __name__ == "__main__":
  main()