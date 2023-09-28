# sampling data for rlhf

after training a model, you need to load it and sample responses from it, 2 per prompt (although you can sample more). And then you need to use alpaca farm to rank them. The best and worst ranking (or may be we should parameterize it?) should be used as `chosen` and `rejected` responses to train a reward model in the next training stage.

`main.py` helps sampling from a model. see [./sample_grainte_13b.sh](./sample_granite_13b.sh) for an example.

importantly, the data_split on this stage assumes `training phase 3`, since this phase makes the data collator only query the prompts of a dataset and nothing else.