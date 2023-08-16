# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
from utils.utils import print_rank_0, Fore
# DeepSpeed Team
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
                        tb_path="",
                        tb_name=""):

    device = "cpu" if offload else "none"
    print_rank_0("remember to get ds_utils stage 3 to manual", color=Fore.GREEN)
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        # "stage3_param_persistence_threshold": 1e9,
        # "stage3_max_live_parameters": 14e9,
        # "stage3_prefetch_bucket_size": 1e8,
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_prefetch_bucket_size": "auto",
        # "memory_efficient_linear": False,
        # "stage3_param_persistence_threshold": 1e6,
        # "stage3_max_live_parameters": 1e9,
        # "stage3_prefetch_bucket_size": 5e8,
          "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": false,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": false
    }
    # "fp16": {
    #     "enabled": True,
    #     "loss_scale_window": 100
    # },
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        # "bf16": {
        #     "enabled": True
        # },
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
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
