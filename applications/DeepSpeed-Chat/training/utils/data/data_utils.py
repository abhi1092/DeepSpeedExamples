# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
from concurrent.futures import ProcessPoolExecutor
import glob
import math
from pathlib import Path
import time
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain, repeat, starmap
from . import raw_datasets
from utils.utils import get_caller, load_hf_tokenizer, print_rank_0


def get_raw_dataset(dataset_name, output_path, seed, local_rank, column_names=None):

    if "Dahoas/rm-static" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "yitingxie/rlhf-reward-datasets" in dataset_name:
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "openai/webgpt_comparisons" in dataset_name:
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "stanfordnlp/SHP" in dataset_name:
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "pvduy/sharegpt_alpaca_oa_vicuna_format" in dataset_name:
        return raw_datasets.PvduySharegptalpacaoavicunaformatDataset(
            output_path, seed, local_rank, dataset_name)
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "mkqa-Chinese" in dataset_name:
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank,
                                               "mkqa")
    elif "mkqa-Japanese" in dataset_name:
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank,
                                                "mkqa")
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "lmqg/qg_jaquad" in dataset_name:
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "lmqg/qag_jaquad" in dataset_name:
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank,
                                                 dataset_name)
    elif "local/jsonfile" in dataset_name:
        chat_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir,
                         os.path.pardir, os.path.pardir))
        if not (os.path.isfile(chat_path + '/data/train.json')
                and os.path.isfile(chat_path + '/data/eval.json')):
            raise RuntimeError(
                f"Please check both the train.json and eval.json files in your applications/DeepSpeed-Chat/data directory."
            )
        return raw_datasets.LocalJsonFileDataset(output_path, seed, local_rank,
                                                 dataset_name, chat_path)
    elif ".jsonl" in Path(dataset_name).name:
        print_rank_0(f'jsonl dataset {dataset_name}', color="GREEN", rank=local_rank)
        return raw_datasets.JsonlDataset(output_path, seed, local_rank, dataset_name, column_names)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_split, split_index,
                                data_size):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    print_rank_0(f"loading dataset index_file_name = {index_file_name}", color="CYAN")
    # reindex each time when using local jsonfile since it's more likely to get modified
    if (not os.path.isfile(index_file_name)) or (dataset_name == 'jsonfile'):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]\
                    if "labels" not in self.chosen_dataset[idx] \
                    else self.chosen_dataset[idx]["labels"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id
    def get_subset(self, indices):
        # Extract subsets from internal datasets if they're not empty
        prompt_subset = [self.prompt_dataset[i] for i in indices] \
            if len(self.prompt_dataset)>0 else []
        chosen_subset = [self.chosen_dataset[i] for i in indices] \
            if len(self.chosen_dataset)>0 else []
        reject_subset = [self.reject_dataset[i] for i in indices] \
            if len(self.reject_dataset)>0 else []

        # Return new instance of PromptDataset
        return PromptDataset(
            prompt_subset,
            chosen_subset,
            reject_subset,
            self.pad_token_id,
            self.train_phase
        )

def process_single_data_point(tmp_data, raw_dataset=None, train_phase=None, tokenizer=None, end_of_conversation_token=None, max_seq_len=None, eos_token_id=None):
    # If the function arguments are None, we're in parallel mode and should use the global variables
    if raw_dataset is None:
        raw_dataset = g_raw_dataset
    if train_phase is None:
        train_phase = g_train_phase
    if tokenizer is None:
        tokenizer = g_tokenizer
    if end_of_conversation_token is None:
        end_of_conversation_token = g_end_of_conversation_token
    if max_seq_len is None:
        max_seq_len = g_max_seq_len
    if eos_token_id is None:
        eos_token_id = g_eos_token_id
    #print everything 
    # print(f"tmp_data = {tmp_data}")
    # print(f"raw_dataset = {raw_dataset}")
    # print(f"train_phase = {train_phase}")
    # print(f"train_phase is 1 = {train_phase==1}")
    # print(f"tokenizer = {tokenizer}")
    # print(f"end_of_conversation_token = {end_of_conversation_token}")
    # print(f"max_seq_len = {max_seq_len}")
    # print(f"eos_token_id = {eos_token_id}")
    if train_phase == 1:
        chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data)
        # print(f"chosen_sentence = {chosen_sentence}")
        prompt = raw_dataset.get_prompt(tmp_data)
        if chosen_sentence is not None:
            # print(f"chosen_sentence is not None")
            chosen_sentence += end_of_conversation_token
            # print(f"chosen_sentence = {chosen_sentence} + {get_caller()}")
            chosen_token = tokenizer(chosen_sentence,
                                    max_length=max_seq_len,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")
            # print(f"chosen_token = {chosen_token['input_ids']}")
            # print(f"eos in chosen_token = {eos_token_id in chosen_token['input_ids'].squeeze(0)}")
            # check if eos anywhere in chosen_token
            if eos_token_id not in chosen_token["input_ids"].squeeze(0):
                return None, None, None
            prompt_token = tokenizer(prompt, 
                                    return_tensors="pt",
                                    max_length=max_seq_len,
                                    padding="max_length",
                                    truncation=True)

            chosen_token['labels'] = chosen_token["input_ids"].clone().squeeze(0)
            # get length of prompt token
            prompt_length = prompt_token["attention_mask"].sum().item()
            chosen_token['labels'][:prompt_length] = -100
            chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                0)
            chosen_token["attention_mask"] = chosen_token[
                "attention_mask"].squeeze(0)
            # print(f"chosen_token = {chosen_token}")
            return chosen_token, None, None
    else:
        # print(f"train_phase_error {train_phase!=1}")
        raise NotImplementedError
    return None, None, None

def data_processing_initializer(_raw_dataset, _train_phase, _tokenizer, _end_of_conversation_token, _max_seq_len, _eos_token_id):
    global g_raw_dataset, g_train_phase, g_tokenizer, g_end_of_conversation_token, g_max_seq_len, g_eos_token_id
    g_raw_dataset = _raw_dataset
    g_train_phase = _train_phase
    g_tokenizer = _tokenizer #load_hf_tokenizer(_tokenizer, fast_tokenizer=False)
    g_end_of_conversation_token = _end_of_conversation_token
    g_max_seq_len = _max_seq_len
    g_eos_token_id = _tokenizer.encode(_end_of_conversation_token)[-1]

def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len, parallel=False):
    print_rank_0(f"Creating dataset", color="RED", include_caller=True)
    start_time = time.time()
    # eos_token_id = tokenizer.encode(end_of_conversation_token)[-1]
    if parallel:
        print_rank_0("Using parallel processing", color="CYAN")
        with ProcessPoolExecutor(max_workers=os.cpu_count(),
                                 initializer=data_processing_initializer, initargs=(raw_dataset,
                                                                                    train_phase,
                                                                                    # tokenizer.name_or_path,
                                                                                    tokenizer,
                                                                                    end_of_conversation_token,
                                                                                    max_seq_len,
                                                                                    # eos_token_id
                                                                                    )) as executor:
            results = list(executor.map(process_single_data_point, current_dataset))
    else:
        args = zip(current_dataset,
                repeat(raw_dataset),
                repeat(train_phase),
                repeat(tokenizer),
                repeat(end_of_conversation_token),
                repeat(max_seq_len),
                repeat(eos_token_id))
        results = list(starmap(process_single_data_point, args))
    
    chosen_dataset, reject_dataset, prompt_dataset = [], [], []    
    if train_phase == 1:
        from IPython import embed; embed(header=get_caller())
        chosen_dataset = [d[0] for d in results if d[0] is not None]
        print_rank_0(f"Number of dropped samples: {len(results) - len(chosen_dataset)}", color="GREEN")
    else:
        raise NotImplementedError
    print_rank_0(f"Time to create dataset: {time.time() - start_time}", color="GREEN", include_caller=True)
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset_split_(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    print_rank_0(f"Creating dataset for {get_caller()}", color="RED", include_caller=True)
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    eos_enc = tokenizer.encode(end_of_conversation_token)
    number_droped = 0
    start_time = time.time()
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            prompt = raw_dataset.get_prompt(tmp_data)
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                #check if eos anywhere in chosen_token
                if eos_enc[-1] not in chosen_token["input_ids"].squeeze(0):
                    number_droped += 1
                    continue
                prompt_token = tokenizer(prompt, 
                                         return_tensors="pt",
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True)
                assert tokenizer.padding_side == "right"
                chosen_token['labels'] = chosen_token["input_ids"].clone().squeeze(0)
                #get lenght of prompt token
                prompt_length = prompt_token["attention_mask"].sum().item()
                chosen_token['labels'][:prompt_length] = -100
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                chosen_dataset.append(chosen_token)
        print_rank_0(f"Number of dropped samples: {number_droped}", color="GREEN")

    elif train_phase == 2:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]
                chosen_dataset.append(chosen_token)

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                reject_dataset.append(reject_token)

    elif train_phase == 3:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                prompt_token = tokenizer(prompt, return_tensors="pt")
                prompt_token["input_ids"] = prompt_token["input_ids"]
                prompt_token["attention_mask"] = prompt_token["attention_mask"]
                for key_word in ["input_ids", "attention_mask"]:
                    length = prompt_token[key_word].size()[-1]
                    if length > max_seq_len:
                        y = prompt_token[key_word].squeeze(0)[length -
                                                              (max_seq_len -
                                                               1):].flip(0)
                    else:
                        y = prompt_token[key_word].squeeze(0).flip(0)
                    prompt_token[key_word] = y
                prompt_dataset.append(prompt_token)
    print_rank_0(f"Time to create dataset: {time.time() - start_time}", color="GREEN", include_caller=True)
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, column_names=None):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank, column_names)
    
    train_dataset = raw_dataset.get_train_data()
    print_rank_0(f"debbuging len(train_dataset) = {len(train_dataset)}", color="RED", include_caller=True)
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset

def save_dataset_splits(dataset, max_num_per_split, file_name):
    #remove all existing files that start with file_name and end in `_i.pt` in the directory of file_name
    for f in os.listdir(os.path.dirname(file_name)):
        if f.startswith(os.path.basename(file_name)) and f.endswith(".pt"):
            print_rank_0(f"Removing {f} data split", color="GREEN", rank=0)
            os.remove(os.path.join(os.path.dirname(file_name), f))
    
    splits = []
    curr = 0
    while curr < len(dataset):
        split_name = f"{file_name}_{curr}.pt"
        splits.append(split_name)
        indices = range(curr, min(curr + max_num_per_split, len(dataset)))
        split_subset = PromptDataset.get_subset(dataset, indices)
        print_rank_0(f"len(split_subset) = {len(split_subset)}", color="YELLOW", rank=0)
        print_rank_0(f"Saving {split_name} data split", color="GREEN", rank=0)
        torch.save(split_subset, split_name)
        curr += max_num_per_split
    return splits


def get_dataset_splits(file_name):
    # Use glob to get all files that start with file_name and end with .pt
    splits = sorted(glob.glob(f"{file_name}_*.pt"))
    
    # If no such files are found, return a list with just the file_name
    if len(splits) == 0:
        splits.append(file_name)
    
    print_rank_0(f"found {splits} data splits", color="MAGENTA")
    return splits

def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[],
                          reload=False,
                          column_names=None, #added only for sampling
                          #set max num split to the maximum integer
                          max_num_per_split=float("inf")
                          ):
    """
    Creates the prompt dataset
    """
    output_path = str(Path(output_path) / f"rank_{os.environ['GROUP_RANK']}")
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}_mnps{max_num_per_split}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    if cache_found:
        print_rank_0(f"Loading cached dataset from {output_path}", color="GREEN")
    else:
        print_rank_0(f"Creating dataset at {output_path}", color="GREEN")
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    if not (buf_create_cache.item() != 0 or reload):
        print_rank_0(f"Loading cached dataset from {train_fname} rank: {local_rank}", color="GREEN")
    else:
        print_rank_0(f"Creating dataset at {train_fname}  rank: {local_rank}", color="GREEN")
        
    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank, data_path[0], data_split, output_path, train_phase,
                seed, tokenizer, end_of_conversation_token, max_seq_len, column_names)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                train_dataset, eval_dataset = create_dataset(
                    local_rank, d_path, data_split, output_path, train_phase,
                    seed, tokenizer, end_of_conversation_token, max_seq_len, column_names)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        if train_phase == 1 and sft_only_data_path:
            sft_train_datasets = []
            sft_eval_datasets = []
            sft_train_size = 0
            sft_eval_size = 0
            for sft_path in sft_only_data_path:
                sft_train_dataset, sft_eval_dataset = create_dataset(
                    local_rank,
                    sft_path,
                    "10,0,0",
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                    column_names,
                )
                sft_train_datasets.append(sft_train_dataset)
                sft_eval_datasets.append(sft_eval_dataset)
                sft_train_size += len(sft_train_dataset)
                sft_eval_size += len(sft_eval_dataset)
            if sft_train_datasets:  # Check if sft_train_datasets is not empty
                sft_train_dataset = ConcatDataset(sft_train_datasets)
                train_dataset = ConcatDataset(
                    [train_dataset, sft_train_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
                train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
                sft_eval_dataset = ConcatDataset(sft_eval_datasets)
                eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
                eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        print_rank_0(f"the number of data in train_dataset: {len(train_dataset)}", color="GREEN")
        print_rank_0(f"Saving dataset to {train_fname} rank: {local_rank}", color="GREEN", rank=0)
        start = time.time()
        train_splits = [train_fname]
        if max_num_per_split < len(train_dataset):
            print_rank_0(f"Splitting train dataset into {math.ceil(len(train_dataset)/max_num_per_split)} splits", color="GREEN", rank=0)
            train_splits = save_dataset_splits(train_dataset, max_num_per_split, train_fname)
        else:
            torch.save(train_dataset, train_fname)
        print_rank_0(f"Time to save train dataset: {time.time() - start}", color="GREEN", rank=0)
        torch.save(eval_dataset, eval_fname)
        #save len_train_dataset to a file
        len_train_dataset = len(train_dataset)
        torch.save(len_train_dataset, f"{output_path}/len_train_dataset_{fname}.pt")
        print_rank_0(f"the number of data in training: {len(train_dataset)}", color="GREEN")
        
    torch.distributed.barrier()
    
    train_splits = get_dataset_splits(train_fname)

    len_train_dataset = torch.load(f"{output_path}/len_train_dataset_{fname}.pt")
    print_rank_0(f"the number of data in training: {len_train_dataset}", color="BLUE", include_caller=True, rank=0)
    
    return train_splits, eval_fname, len_train_dataset


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(0, pad_length),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []
