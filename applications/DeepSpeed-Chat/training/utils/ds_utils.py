# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
from utils.utils import print_rank_0, Fore
# DeepSpeed Team

import torch

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):

    device = "cpu" if offload else "none"
    print_rank_0("remember to get ds_utils stage 3 to manual", color=Fore.GREEN)
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device,
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": device,
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "stage3_max_live_parameters": 14e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_prefetch_bucket_size": 1e9,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_gather_16bit_weights_on_model_save": False,
        "sub_group_size": 1e12,
        # "stage3_param_persistence_threshold": 1e9,
        # "stage3_max_live_parameters": 14e9,
        # "stage3_prefetch_bucket_size": 1e8,
        # "stage3_param_persistence_threshold": "auto",
        # "stage3_max_live_parameters": "auto",
        # "stage3_prefetch_bucket_size": "auto",
        "memory_efficient_linear": True,
        # "stage3_param_persistence_threshold": 1e6,
        # "stage3_max_live_parameters": 1e9,
        # "stage3_prefetch_bucket_size": 5e8,
        #   "allgather_partitions": True,
        # "allgather_bucket_size": 1e9,
        # "overlap_comm": False,
        # "reduce_scatter": True,
        # "reduce_bucket_size": 1e8,
        # "contiguous_gradients": False
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count()
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        # "fp16": {
        #     "enabled": True,
        #     "loss_scale_window": 100
        # },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": True,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        },
        # "flops_profiler": {
        #     "enabled": True,
        #     "profile_step": 0,
        #     "module_depth": -1,
        #     "top_modules": 1,
        #     "detailed": True,
        #     "output_file": None
        # }
    }


def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        # "bf16": {
        #     "enabled": True
        # },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
    
def get_inference_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e9,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": True,
    }
    return {
        "zero": zero_opt_dict,
        # "fp16": {
        #     "enabled": True
        # },
        # "bf16": {
        #     "enabled": True
        # },
        "tensor_parallel": {
            "tp_size": 1,
        },
        "enable_cuda_graph": True,
        # "wall_clock_breakdown": True
    }


def get_ds_config():
    return {
     "bf16": {
            "enabled": True
        },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },


    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}
