# %%
import numpy as np
import pandas as pd
import json
# %%
# opening smple prompts
sample_prompts = pd.read_csv('datasets\prompt_samples.csv')
print(sample_prompts.shape)
print(sample_prompts.head())
# %%
# opening data in json format and utf-8 encoding
with open('datasets\doccano_cause_effec_revision_1.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
# %%
non_causal_count = 0

for item in data:
    if item['relations']== []:
        non_causal_count += 1
print(f"Number of non-causal samples: {non_causal_count}")
# %%
