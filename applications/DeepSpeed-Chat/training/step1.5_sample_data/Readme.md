# sampling data for rlhf

after training a model, you need to load it and sample responses from it, 2 per prompt (although you can sample more). And then you need to use alpaca farm to rank them. The best and worst ranking (or may be we should parameterize it?) should be used as `chosen` and `rejected` responses to train a reward model in the next training stage.