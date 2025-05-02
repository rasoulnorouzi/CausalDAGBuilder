#%%
"""
llama3_causality_smoketest_inline_token.py
-----------------------------------------
Put your actual access token in HF_TOKEN.  
Run:  python llama3_causality_smoketest_inline_token.py
"""

import json
from huggingface_hub import login
from vllm import LLM, SamplingParams

# >>>>> 1. inline token (replace the string) <<<<<
HF_TOKEN = "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"
login(token=HF_TOKEN)

# 2. config that matches your note
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SEED       = 8642
MAX_LEN    = 512

llm = LLM(model=MODEL_NAME,
          seed=8642,
          max_model_len=512)

# 3 ▪ sentence to test
demo_sentence = "Because the interest rate rose, housing demand fell sharply."

# 4 ▪ conversation in **chat-template** format  ────────────────────────────────
conversation = [
    {"role": "system",
     "content":
        "You are an assistant that decides whether the given sentence "
        "expresses a causal relation. "
        "If NON-CAUSAL, reply exactly NON-CAUSAL. "
        "If CAUSAL, reply with a ONE-LINE JSON object "
        'like {"cause":"…","effect":"…","polarity":"positive|negative"}. '
        "Do not output anything else."},
    {"role": "user", "content": demo_sentence},
]

# 5 ▪ run the model via the chat helper (auto-templates prompt)  ───────────────
out = llm.chat(
        conversation,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=64)
)

# 6 ▪ print the raw assistant content
# The vllm library returns output in a different structure
# Use the appropriate attribute to get the model's response
print(out[0].outputs[0].text.strip())
################