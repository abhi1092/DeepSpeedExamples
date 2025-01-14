# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import sys
import os
import time
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import print_rank_0, get_caller, Fore
from utils.consts import OASST_PROMPT, STOP_KEYS


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.reward_tokenizer = self.rlhf_engine.reward_tokenizer
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.generate_time = 0.0

    def _generate_sequence(self, prompts, mask, step):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]
        
        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        if self.actor_model.model.config.model_type == "llama":
            kwargs = dict(do_sample=False)
        else:
            kwargs = dict()

        with torch.no_grad():
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled,
                repetition_penalty=1.2,
                temperature=.7,
                do_sample=True,
                top_p=.9,
                top_k=50,
                **kwargs,
                )
        #print seq.shape
        print_rank_0(f"seq: {seq.shape}", self.args.local_rank, color=Fore.GREEN)
        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        
        if type(self.rlhf_engine.reward_tokenizer) != type(self.rlhf_engine.tokenizer):
            print_rank_0("ppo_trainer.py:93 it's not enough to simply check if the tokenizers are different, because granite requires a treatment that is not the same for everyone else.", color=Fore.RED)
            print_rank_0(f"prompts {prompts.shape}", color=Fore.GREEN)
            
            prompts_str = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)
            rm_prompts = self.reward_tokenizer(prompts_str, max_length=self.args.max_prompt_seq_len,
                                                            padding="max_length",
                                                            truncation=True,
                                                            return_tensors="pt")['input_ids'].to("cuda")
            print_rank_0(f"rm_prompts {rm_prompts.shape}", color=Fore.GREEN, rank=torch.distributed.get_rank())
            ans_str = self.tokenizer.batch_decode(ans)
            
            #check if any of the stop keys are in ans_str
            print_rank_0(f"STOP_KEYS in ans {[any(s in a for s in STOP_KEYS) for a in ans_str]}", color=Fore.GREEN, rank=torch.distributed.get_rank())
            
            #find first index of any stop key
            end_idx = [next((a.index(s) for s in STOP_KEYS if s in ans_str), -1) for a in ans_str]
                                         
            #remove everything after the end key and add the end of text token.
            ans_str = [a[:e] + "<|endoftext|>" for a,e in zip(ans_str, end_idx)]
            # rm_input = [OASST_PROMPT.format(instruction=p.replace("<|endoftext|>", ""), 
            #                                 response=a.replace("<|endoftext|>","")) 
            #             for p,a in zip(prompts_str, ans_str)]
            rm_input = self.reward_tokenizer([p + a for p,a in zip(prompts_str, ans_str)], max_length=self.args.max_prompt_seq_len + self.args.max_answer_seq_len,
                                                                                            padding="max_length",
                                                                                            truncation=True,
                                                                                            return_tensors="pt").to("cuda")
            print_rank_0(f"rm_input {rm_input['input_ids'].shape}", color=Fore.GREEN)            
        else:
            rm_input = {"input_ids": seq, "attention_mask": seq.not_equal(self.tokenizer.pad_token_id).long()}
            rm_prompts = prompts
        
        if self.args.print_answers and torch.distributed.get_rank() == 0:
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        rm_input_ids, rm_attn_mask = [], []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
                rm_input_ids.append(rm_input["input_ids"][i:i + 1])
                rm_attn_mask.append(rm_input["attention_mask"][i:i + 1])
                
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return out_seq, rm_input, rm_prompts #and return another one.

    def generate_experience(self, prompts, mask, step):
        #TODO: I need to use both the prompts and the formatted prompts.
        self.eval()
<<<<<<< HEAD
        seq, rm_input, rm_prompts = self._generate_sequence(prompts, mask, step)
=======
        generate_start = time.time()
        seq = self._generate_sequence(prompts, mask, step)
        generate_end = time.time()
>>>>>>> fixing_oom_step0
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            reward_score = self.reward_model.forward_value(
                **rm_input,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
            values = self.critic_model.forward_value(
                **rm_input, return_value_only=True).detach()[:, :-1] #TODO: why the :-1?

        logits = output.logits
        logits_ref = output_ref.logits
<<<<<<< HEAD
    
=======

        self.generate_time = generate_end - generate_start

>>>>>>> fixing_oom_step0
        return {
            'prompts': prompts,
            'rm_prompts': rm_prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask,
            'rm_input_ids': rm_input["input_ids"],
            'rm_attention_mask': rm_input["attention_mask"],
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        #TODO: so the rewards end up being the same length as the actions and then we add a reward to those (which make sense)
        # from IPython import embed; embed(header=get_caller())
        #print logprobs
        print_rank_0(f"log_probs: {log_probs.shape}", self.args.local_rank, color=Fore.GREEN)
        #print ref_log_probs
        print_rank_0(f"ref_log_probs: {ref_log_probs.shape}", self.args.local_rank, color=Fore.GREEN)
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        #print kl_divergence_estimate
        print_rank_0(f"kl_divergence_estimate: {kl_divergence_estimate.shape}", self.args.local_rank, color=Fore.GREEN)
        
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        print_rank_0(f"rm_action_mask: {action_mask.shape, action_mask.sum(1)}", self.args.local_rank, color=Fore.GREEN)
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        #print reward clip value and shape
        print_rank_0(f"reward_clip: {reward_clip}", self.args.local_rank, color=Fore.GREEN)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j] #reward is being put only at the end of the sequence!!!!

        return rewards

    def train_rlhf(self, inputs):
        # from IPython import embed; embed(header=get_caller())
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        rm_prompts = inputs['rm_prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards'] #so what's up with the reward score???.
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']
        rm_seq = inputs['rm_input_ids']
        rm_mask = inputs['rm_attention_mask']
        # from IPython import embed; embed(header=get_caller())
        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]
        rm_action_mask = rm_mask[:, 1:]

        # from IPython import embed; embed(header=get_caller())
        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(rm_prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               rm_action_mask)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(input_ids=rm_seq,
                                                attention_mask=rm_mask,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:, 
                                                                       start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        return actor_loss, critic_loss

    def get_overflow(self):
        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        # TODO: entropy regularization
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        # from IPython import embed; embed(header=get_caller())
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        # from IPython import embed; embed(header=get_caller())
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        print_rank_0(f"rewards shape: {rewards.shape}", self.args.local_rank, color=Fore.GREEN)
        print_rank_0(f"start: {start}", self.args.local_rank, color=Fore.GREEN)
        print_rank_0(f"length: {length}", self.args.local_rank, color=Fore.GREEN)        
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
