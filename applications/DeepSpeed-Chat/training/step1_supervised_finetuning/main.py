#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
from datetime import timedelta
import os
import math
import sys
import time
import optuna

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import get_caller, get_column_names, print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from utils.model.model_utils import create_hf_model
from utils.perf import print_throughput


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--reload_data',
                        action='store_true',
                        help='Reloads the data even if it was already preprocessed.')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument('--max_num_per_split',
                        type=int,
                        default=int(1.7e5),
                        help='saves splits of the dataset of maximum this size and loads them to memory sequentially to avoid memory issues.')
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
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--save_checkpoint",
                        action='store_true',
                        help="Save a deepspeed checkpoint every save_steps steps.")
    parser.add_argument("--load_checkpoint_path",
                        type=str,
                        default=None,
                        help="Path to load checkpoint from.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=3000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    
    ## optuna optimization
    parser.add_argument("--optuna_study_name",
                        type=str,
                        default=None,
                        help="The study name for optuna optimization.")
    parser.add_argument("--optuna_storage",
                        type=str,
                        default=None,
                        help="The storage for optuna optimization.")
    parser.add_argument("--max_time",
                        type=int,
                        default=None,
                        help="The max number of seconds for optuna optimization.")
    
    
    parser = deepspeed.add_config_arguments(parser)
    
    
    args = parser.parse_args()

    return args

def setup_optuna(args):
    study = None
    trial = None
    optuna_start_time = None
    if args.optuna_study_name and args.optuna_storage and args.local_rank == 0:
        study = optuna.load_study(study_name=args.optuna_study_name, storage=args.optuna_storage)
        trial = study.ask()
        print(f"TRIAL_NUMBER:{trial.number}", flush=True)
        optuna_start_time = time.time()
        args.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 0.1, log=True)
        if args.weight_decay <= 1e-5:
            args.weight_decay = 0.0
    return study, trial, optuna_start_time

def process_data(args, tokenizer, end_of_conversation_token):
    '''
    This function will create the training and evaluation dataloaders.
    
    '''
    # Prepare the data
    train_phase = 1
    train_splits, eval_fname, len_train_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        sft_only_data_path=args.sft_only_data_path,
        column_names=args.column_names,
        end_of_conversation_token=end_of_conversation_token,
        reload=args.reload_data,
        max_num_per_split=args.max_num_per_split,
    )
    print_rank_0(f"len_train_dataset: {len_train_dataset}", color="GREEN")
    start = time.time()
    print_rank_0(f"loading train_splits: {train_splits[0]} of {train_splits}", color="GREEN")
    train_dataset = torch.load(train_splits[0])
    eval_dataset = torch.load(eval_fname)
    print_rank_0(f"Loading data took {time.time() - start} seconds", color="GREEN")
    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    yield train_dataloader, eval_dataloader, len_train_dataset
    
    # keep yielding the next training splits
    for split in train_splits[1:]:
        print_rank_0(f"loading train_splits: {split}", color="GREEN")
        train_dataset = torch.load(split)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      collate_fn=default_data_collator,
                                      sampler=train_sampler,
                                      batch_size=args.per_device_train_batch_size)
        yield train_dataloader

def save_model_operations(model, tokenizer, args, epoch, step):
    if args.output_dir is None:
        return

    if (step + 1) % args.save_steps == 0:
        if args.save_checkpoint:
            output_path = os.path.join(args.output_dir, "deepspeed_checkpoint")
            os.makedirs(output_path, exist_ok=True)
            model.save_checkpoint(output_path)

        print_rank_0('saving model ...', args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, sub_folder=f"step1_model/epoch_{epoch}_step_{step}")

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model, args.global_rank, args.output_dir, zero_stage=args.zero_stage)

def main():
    args = parse_args()
    args.column_names = get_column_names(args)
    
    args.local_rank = int(os.environ["LOCAL_RANK"])
    
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed(timeout=timedelta(minutes=300))
        
    study, trial, optuna_start_time = setup_optuna(args)

    args.global_rank = torch.distributed.get_rank()
    print_rank_0(f'column_names: {args.column_names}', color="GREEN")
    print_rank_0(f"args: {args}", color="GREEN")
    print_rank_0(f"ENV: {os.environ}", color="GREEN")
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
   
    HUMAN_KEY = "<|user|>"
    ASSISTANT_KEY = "<|assistant|>"
    CONTEXT_KEY = "<|context|>"
    END_KEY = "<|end|>"
   
    tokenizer.add_special_tokens({"additional_special_tokens": [CONTEXT_KEY, HUMAN_KEY, ASSISTANT_KEY, END_KEY]})
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            disable_dropout=args.disable_dropout)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)
            
    data_generator = process_data(args, tokenizer, end_of_conversation_token=END_KEY)
    train_dataloader, eval_dataloader, len_train_dataset = next(data_generator)
    
    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        step = 1
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        model.train()
        return perplexity
    

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len_train_dataset / args.gradient_accumulation_steps)
    if args.num_warmup_steps == -1:
        args.num_warmup_steps = math.ceil(num_update_steps_per_epoch * args.num_train_epochs * 0.03)
        print_rank_0(f"num_warmup_steps: {args.num_warmup_steps}", color="YELLOW")
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    #load checkpoint of load_checkpoint_path is not none and it contains a checkpoint
    if args.load_checkpoint_path and os.path.exists(args.load_checkpoint_path):
        print_rank_0(f"Loading checkpoint from {args.load_checkpoint_path}", color="GREEN")
        model.load_checkpoint(args.load_checkpoint_path)
        torch.distributed.barrier()
        
    def optuna_operations(loss, step, final=False):
        if study:
            print_rank_0(f"debugging", color="RED", include_caller=True)
            if final:
                # Report final metric
                print_rank_0(f"debugging", color="RED", include_caller=True)
                final_metric_value = evaluation(model, eval_dataloader)
                print_rank_0(f"debugging", color="RED", include_caller=True)
                study.tell(trial, final_metric_value)
            else:
                if args.global_rank == 0:
                    trial.report(loss.item(), step=step)
                    print_rank_0(f"debugging", color="RED", include_caller=True)
                    # Pruning based on the loss
                    print_rank_0(f"debugging", color="RED", include_caller=True)
                    if trial.should_prune():
                        print_rank_0(f"debugging", color="RED", include_caller=True)
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        exit()
                    
            # Check if max_time has passed and perform evaluation
            elapsed_time = time.time() - optuna_start_time
            if args.max_time and elapsed_time > args.max_time:
                print_rank_0(f"debugging", color="RED", include_caller=True)
                print_rank_0(f"EVALUATING AFTER MAX TIME EXCEDEED", color="GREEN")
                final_metric_value = evaluation(model, eval_dataloader)
                if args.global_rank == 0:
                    study.tell(trial, final_metric_value)
                print(f"Max time exceeded. Final Metric Value: {final_metric_value}")
                exit()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    for epoch in range(args.num_train_epochs):
        step = 0
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        while train_dataloader is not None:
            for batch in train_dataloader:
                start = time.time()
                    
                batch = to_device(batch, device)
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                if args.print_loss:
                    print(
                        f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                    )
                model.backward(loss)
                model.step()
                end = time.time()
                if torch.distributed.get_rank() == 0:
                    print_throughput(model.module, args, end - start,
                                    args.global_rank)
                if step % 5 == 0:
                    optuna_operations(loss, step)
                
                save_model_operations(model, tokenizer, args, epoch, step)
                
                step += 1
            
            train_dataloader = next(data_generator, None)

        if args.save_checkpoint:
            output_path = os.path.join(args.output_dir, "deepspeed_checkpoint")
            os.makedirs(output_path, exist_ok=True)
            model.save_checkpoint(output_path)

        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
        
        save_model_operations(model, tokenizer, args, epoch, step)
        torch.distributed.barrier()
        
        #check if not last epoch, if yes, start the train loader again
        if epoch != args.num_train_epochs - 1:
            data_generator = process_data(args, tokenizer, end_of_conversation_token=END_KEY)
            train_dataloader, _, _ = next(data_generator)



if __name__ == "__main__":
    main()
