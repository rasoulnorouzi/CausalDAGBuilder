from huggingface_hub import login
from vllm import LLM, SamplingParams
import json, itertools, gc, os, sys
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
from transformers import AutoTokenizer


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = "your_access_token_here"
INPUT_CSV     = "sentences.csv"     # must contain a 'sentence' column
OUTPUT_CSV    = "annotated.csv"     # will be (over)written
PROMPT_FILE   = "prompt.txt"        # must contain {{SENTENCE}}
BATCH_SIZE    = 8                   # safe for 24-GB A10
MAX_MODEL_LEN = 512
SEED          = 8642
TEMPERATURE   = 0.0
MAX_TOKENS   = 64                  # max tokens to generate
TOP_P        = 0.95                # nucleus sampling
TOP_K        = 40                  # top-k sampling
CHUNK_ROWS    = 10_000              # pandas chunk size (keeps RAM small)



llm = LLM(model=MODEL_NAME,
          seed=SEED,
          max_model_len=MAX_MODEL_LEN,
          sampling_params=SamplingParams(temperature=TEMPERATURE, 
                                         top_p=TOP_P,
                                         top_k=TOP_K,
                                         max_tokens=MAX_TOKENS))



#%% Define test sentence
df_sentences = pd.read_csv(r"C:\Users\norouzin\Desktop\CausalDAGBuilder\weak_supervision\dataset\100k_causal_sentences.csv")
df_sentences.head()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# 6 ▪ PREPARE PROMPT TEMPLATE
template_text = Path("weak_supervision/prompt.txt").read_text()
assert "{{SENTENCE}}" in template_text, "prompt.txt must contain {{SENTENCE}}"

def make_prompt(sentence: str) -> str:
    """Fill the placeholder and wrap with Llama-3 chat template."""
    sys_block = template_text.replace("{{SENTENCE}}", sentence)
    messages  = [{"role": "system", "content": sys_block}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

sampler = SamplingParams(temperature=config.TEMPERATURE,
                         max_tokens=256,
                         stop=["</s>"])

# 7 ▪ HELPER: micro-batch iterator
def micro_batch_iter(iterable, batch_size=10):
    """Yield micro-batches of the given size."""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch

