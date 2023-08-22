import torch
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
  default_data_collator,
)
import argparse
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, save_zero_three_model, load_hf_tokenizer
from sampling_engine import SamplingEngine

def parse_args():
  parser = argparse.ArgumentParser(
    description="(Step 1.5) sample data from the SFT model to collect human evaluations"
  )
  
  parser.add_argument(
      '--data_path',
      nargs='*',
      default=['Dahoas/rm-static'],
      help=
      'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
  )
  parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="Path to pre-trained model"
  )
  parser.add_argument(
      '--data_split',
      type=str,
      default='2,4,4',
      help=
      'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
      'will use 60%% of data for phase 1, 20%% for phase 2 and 20%% for phase 3.'
  )
  parser.add_argument(
      '--data_output_path',
      type=str,
      default='/tmp/data_files',
      help=
      'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
  )
  
  parser.add_argument(
      "--per_device_batch_size",
      type=int,
      default=16,
      help=
      "Batch size (per device) for the dataloader and generation purpose."
  )
  parser.add_argument("--max_seq_len",
                      type=int,
                      default=512,
                      help="The maximum sequence length.")
  parser.add_argument("--temperature",
                      type=float,
                      default=0.7,
                      help="The value used to module the next token probabilities.")
  parser.add_argument("--top_k",
                      type=int,
                      default=0,
                      help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
  parser.add_argument("--top_p",
                      type=float,
                      default=0.9,
                      help="The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.")
  parser.add_argument("--repetition_penalty",
                      type=float,
                      default=1.0,
                      help="The parameter for repetition penalty. 1.0 means no penalty")
  parser.add_argument("--num_return_sequences",
                      type=int,
                      default=1,
                      help="The number of samples to generate per prompt.")
  
  parser.add_argument("--output_dir",
                      type=str,
                      default=None,
                      help="Where to store the model.")
  parser.add_argument("--seed",
                      type=int,
                      default=None,
                      help="A seed for reproducible training.")
  parser.add_argument("--local_rank",
                      type=int,
                      default=-1,
                      help="local_rank for distributed training on gpus")
  parser.add_argument("--disable_dropout",
                      action="store_true",
                      help="Disable dropout ")
  parser.add_argument("--zero_stage",
                      type=int,
                      default=0,
                      help="ZeRO stage to use")
  parser.add_argument("--offload",
                      action="store_true",
                      help="Offload model to CPU")

def main():
  args = parse_args()
  args.local_rank = int(os.environ["LOCAL_RANK"])
  
  if args.local_rank == -1:
      device = torch.device("cuda")
  else:
      torch.cuda.set_device(args.local_rank)
      device = torch.device("cuda", args.local_rank)
      # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
      deepspeed.init_distributed()
  
  set_random_seed(args.seed)
  tensor = torch.ByteTensor([False]).cuda()
  torch.distributed.all_reduce(tensor)
  
  print(f"All reduce test 1 on global rank {args.global_rank} rank {args.local_rank}")
  torch.distributed.barrier()

  # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
  tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                fast_tokenizer=True)
  train_phase = 3 #train phase 3 can be used to get only the prompts, data_split should be 0,0,1
  dataset, _ = create_prompt_dataset(
    args.local_rank,
    args.data_path,
    args.data_split,
    args.data_output_path,
    train_phase,
    args.seed,
    tokenizer,
    args.max_seq_len,
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
      
  sampling_engine = SamplingEngine(args.model_name_or_path, tokenizer, args)
  
  for step, batch_prompt in enumerate(dataloader):
    batch_prompt = to_device(batch_prompt, device)
    out = sampling_engine.generate_sequence(batch_prompt['prompt'],
                                            batch_prompt['prompt_att_mask'])
    print(out)