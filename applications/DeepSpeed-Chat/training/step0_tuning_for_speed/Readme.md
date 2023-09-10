# tuning for throughput

One important part of any training pipeline is to tune DeepSpeed hyperparameters. In this pipeline all trainers and samplers are custom, so they don't really work with `deepspeed autotune`. In this step we solve this problem by providing a dummy trainer that can be used to find the best deepspeed parameters for your training hardware.

# Granite 13b

we tuned granite 13b using this command:
```bash
deepspeed --autotuning tune --num_nodes=1 --num_gpus=8 \
 main.py\
    --model_name_or_path /new_data/rl-4-llm/granite_models/pretrained/step_225000_ckpt/\
  --data_path /new_data/datasets/summarization/tldr_sft_train_117k.jsonl \
  --prompt formatted_input\
  --chosen summary\
  --data_split 1,10,10\
  --data_output_path /app/.local_data/tuning \
  --max_seq_len 2048\
  --gradient_checkpointing\
  --seed 1234\
  --per_device_batch_size 1\
  --learning_rate 5e-5\
  --weight_decay 0.01\
  --deepspeed\
  --deepspeed_config ds_config_z3.json
```

importantly, we need gradient checkpointing or a sequence length of 2048 is not feasible. However, offloading was not necessary.

We needed to use a modified version of the autotuner of deepspeed, which is installed by installing the fork we used:

```bash
git clone https://github.com/aldopareja/DeepSpeed.git && \
    cd DeepSpeed && \
    git checkout search_error_fix && \
    TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 DS_BUILD_FUSED_ADAM=1 pip install -v --global-option="build_ext" --global-option="-j8" --no-cache-dir -e .[autotuning_ml,autotuning]
```

The only difference is on searching for errors in `stderr`, the vanilla `DeepSpeed` repo confuses a warning of loading a granite model with gpt-code with an error and then stops the autotuning.

you also need to install `megatron-models` from IBM that enables the use of `granite` models. [repo](github.ibm.com/ai-models-architectures/megatron-models.git)



