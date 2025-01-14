# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
from utils.utils import print_rank_0, Fore
# DeepSpeed Team

import torch
import deepspeed.comm as dist

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
# def get_train_ds_config(offload,
#                         stage=2,
#                         enable_hybrid_engine=False,
#                         inference_tp_size=1,
#                         release_inference_cache=False,
#                         pin_parameters=True,
#                         tp_gather_partition_size=8,
#                         max_out_tokens=512,
#                         enable_tensorboard=False,
#                         enable_mixed_precision_lora=False,
#                         tb_path="",
#                         tb_name=""):

#     device = "cpu" if offload else "none"
#     zero_opt_dict = {
#         "stage": stage,
#         "offload_param": {
#             "device": device
#         },
#         "offload_optimizer": {
#             "device": device
#         },
#         "stage3_param_persistence_threshold": 1e4,
#         "stage3_max_live_parameters": 3e7,
#         "stage3_prefetch_bucket_size": 3e7,
#         "memory_efficient_linear": False
#     }
#     if enable_mixed_precision_lora:
#         zero_opt_dict["zero_quantized_nontrainable_weights"] = True
#         zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count()
#     return {
#         "train_batch_size": GLOBAL_BATCH_SIZE,
#         "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
#         "steps_per_print": 10,
#         "zero_optimization": zero_opt_dict,
#         "fp16": {
#             "enabled": True,
#             "loss_scale_window": 100
#         },
#         "gradient_clipping": 1.0,
#         "prescale_gradients": False,
#         "wall_clock_breakdown": False,
#         "hybrid_engine": {
#             "enabled": enable_hybrid_engine,
#             "max_out_tokens": max_out_tokens,
#             "inference_tp_size": inference_tp_size,
#             "release_inference_cache": release_inference_cache,
#             "pin_parameters": pin_parameters,
#             "tp_gather_partition_size": tp_gather_partition_size,
#         },
#         "tensorboard": {
#             "enabled": enable_tensorboard,
#             "output_path": f"{tb_path}/ds_tensorboard_logs/",
#             "job_name": f"{tb_name}_tensorboard"
#         }
#     }


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
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != torch.cuda.device_count():
            zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count(
            )
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        # "train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        # "fp16": {
        #     "enabled": True,
        #     "loss_scale_window": 100
        # },
        "bf16": {
            "enabled": True
        },
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
        "train_micro_batch_size_per_gpu": "auto",
        "autotuning": {
            "enabled": True,
            "fast": False,
            "arg_mappings": {
            "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
            "gradient_accumulation_steps ": "--gradient_accumulation_steps"
            },
            "results_dir": "outputs/autotuning_results/results",
            "exps_dir": "outputs/autotuning_results/exps"
        }
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
        "tensor_parallel": {
            "tp_size": 4,
        },
        # "enable_cuda_graph": True,
        # "wall_clock_breakdown": True
    }
