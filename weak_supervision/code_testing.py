# %%
"""
llama3_causality_smoketest_inline_token.py
-----------------------------------------
Put your actual access token in HF_TOKEN.  
Run:  python llama3_causality_smoketest_inline_token.py
"""

#%% Import libraries
import json
from huggingface_hub import login
from vllm import LLM, SamplingParams


#%% Configure model
# 2. config that matches your note
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SEED       = 42
MAX_LEN    = 512

llm = LLM(model=MODEL_NAME,
          seed=42,
          max_model_len=512)

#%% Define test sentence
# 3 ▪ sentence to test
demo_sentence = "Because the interest rate rose, housing demand fell sharply."

#%% Create conversation prompt
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

#%% Run model and get response
# 5 ▪ run the model via the chat helper (auto-templates prompt)  ───────────────
out = llm.chat(
        conversation,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=64)
)

#%% Print results
# 6 ▪ print the raw assistant content
# The vllm library returns output in a different structure
# Use the appropriate attribute to get the model's response
print(out[0].outputs[0].text.strip())

# %%
