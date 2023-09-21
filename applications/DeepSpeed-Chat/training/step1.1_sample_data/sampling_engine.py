import deepspeed

import torch

from transformers import AutoModelForCausalLM
from utils.ds_utils import get_inference_ds_config
from utils.model.model_utils import create_hf_model


class SamplingEngine():
  def __init__(self, model_name_or_path, tokenizer, args):
    self.args = args
    self.model_name_or_path = model_name_or_path
    self.tokenizer = tokenizer
    self.model = self._init_model(
     model_name_or_path, args.zero_stage
    )
    
  def _init_model(self, model_name_or_path, zero_stage):
    # DS Config
    ds_config = get_inference_ds_config(self.args.offload,
                                   zero_stage)
    # ds_config[
    #     'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
    # ds_config[
    #     'train_batch_size'] = self.args.per_device_batch_size * torch.distributed.get_world_size()
  
    # HF Model
    model = create_hf_model(
        model_class=AutoModelForCausalLM,
        model_name_or_path=model_name_or_path,
        tokenizer=self.tokenizer,
        ds_config=ds_config,
        disable_dropout=self.args.disable_dropout)

    # initialise Deepspeed ZeRO and store only the engine object
    ds_engine = deepspeed.init_inference(model=model, config=ds_config)
    ds_engine.module.eval()  # inference
    
    return ds_engine
  
  def generate_sequence(self, prompts, mask):
    with torch.no_grad():
      # generate sequences
      generated_sequences = self.model.module.generate(
          input_ids=prompts,
          attention_mask=mask,
          max_length=self.args.max_seq_len,
          temperature=self.args.temperature,
          top_k=self.args.top_k,
          top_p=self.args.top_p,
          repetition_penalty=self.args.repetition_penalty,
          do_sample=True,
          num_return_sequences=self.args.num_answers_per_prompt,
          pad_token_id=self.tokenizer.eos_token_id,
      )
      
    return generated_sequences