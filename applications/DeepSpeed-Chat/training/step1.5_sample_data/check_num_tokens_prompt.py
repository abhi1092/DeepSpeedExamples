'''
This code processes a dataset of prompts using the OpenAssistant/falcon-40b-sft-top1-560 tokenizer from the Hugging Face Transformers library. It loads a dataset from a JSON file, selects the first 100,000 samples, and retrieves the "formatted_input" field from each sample. It then calculates the length of each prompt in tokens using the tokenizer, and filters out the samples where the prompt length is greater than 1600 tokens. The resulting dataset is then subsampled to 30,000 samples and saved to a new JSON file. Finally, the code calculates the quantiles of prompt lengths for the subsampled dataset and prints them to the console.
'''

from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/falcon-40b-sft-top1-560")

data = load_dataset("json", data_files="/new_data/datasets/flan/t0_550k.jsonl", split="train").select(range(100000))

prompts = data["formatted_input"]

# prompts_lengths = tokenizer(prompts, padding="max_length", truncation="longest_first", return_tensors="pt")['attention_mask'].sum(-1)
# prompts_lengths = tokenizer(prompts, padding="max_length", truncation=True, max_length=2048, return_tensors="pt")['attention_mask'].sum(-1)
prompt_lengths = tokenizer(prompts, padding="max_length", truncation=True, max_length=2048, return_tensors="pt")['attention_mask'].sum(-1)

#filter out datapoints where prompt_lengths > 1600
selected_indices = np.where(prompt_lengths <= 1600)[0]
data_filtered = data.select(selected_indices)

#save data_100k_filtered as a jsonl
data_30k = data_filtered.select(range(30000))
with open("/new_data/datasets/flan/t0_30k_filtered.jsonl", "w") as f:
  for sample in data_30k:
    f.write(json.dumps(sample))
    f.write("\n")
    
prompts_lengths_30k = tokenizer(data_30k["formatted_input"], padding="max_length", truncation=True, max_length=2048, return_tensors="pt")['attention_mask'].sum(-1)
quantiles_perc = np.linspace(.9, 1, 11)
num_prompts = np.quantile(prompts_lengths_30k, quantiles_perc)
print("Percentage of prompts with length less than or equal to the quantile")
print("Percentage\tValue")
for i in range(len(quantiles_perc)):
  print(f"{quantiles_perc[i]*100:6.2f}%\t\t{num_prompts[i]:.0f}")
  
  