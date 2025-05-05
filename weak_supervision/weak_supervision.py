# %%
from typing import List, Optional
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import datetime
import os
# %%

"""Simple, self-contained pipeline to annotate sentences with a large-language model
   using vLLM. Designed for a single-GPU workstation (e.g. NVIDIA A10, 24 GB VRAM).

   This version defaults to **Meta-Llama-3-8B-Instruct** hosted on Hugging Face
   (repo: ``meta-llama/Meta-Llama-3-8B-Instruct``). Change ``MODEL_NAME`` below
   if you wish to try a different checkpoint.

   Key features
   ─────────────
   • Reads a prompt template from ``prompt.txt`` that must contain the placeholder
     ``{{SENTENCE}}``.
   • Optionally wraps every prompt with the model-specific *chat template* via
     ``tokenizer.apply_chat_template`` so that BOS/EOS and role tokens exactly
     match the format used during instruction-tuning.
   • Reads sentences from ``100k_causal_sentences.csv`` (first column assumed to
     hold the sentences). Adjust ``DATA_FILE`` or ``SENTENCE_COLUMN`` if needed.
   • Set ``SAMPLE_SIZE`` to an integer (e.g. ``100``) to annotate only that many
     randomly sampled sentences, or ``None`` to process the entire dataset.
   • Stores all model responses in a Python list called ``results`` and prints
     them to ``stdout`` at the end—no files are written.

   Run with: ``python vllm_annotation.py`` (no command-line arguments).
"""

# ── User-tunable parameters ────────────────────────────────────────────────────
MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # default Hugging Face repo
DATA_FILE: str = "/home/rnorouzini/CausalDAGBuilder/weak_supervision/dataset/100k_causal_sentences.csv"            # path to CSV dataset
PROMPT_FILE: str = "/home/rnorouzini/CausalDAGBuilder/weak_supervision/prompt.txt"                         # path to prompt template
SENTENCE_COLUMN: int = 0                                                                # CSV column index
SAMPLE_SIZE: Optional[int] = None   # e.g. 100 -> sample 100 rows, None -> all
USE_CHAT_TEMPLATE: bool = True       # toggle chat wrapping on/off
BATCH_SIZE: int = 1000                # prompts per forward pass
MAX_TOKENS: int = 512                # max new tokens to generate
GPU_MEMORY_UTILISATION: float = 0.90 # fraction of GPU RAM vLLM may allocate
RANDOM_SEED: int = 8642                # reproducible sampling
HF_TOKEN       = "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"
SAVE_INTERVAL: int = 5000            # Save results every N samples processed
# ────────────────────────────────────────────────────────────────────────────────
# %%

def load_prompt_template(path: str) -> str:
    """Read the prompt template containing the ``{{SENTENCE}}`` placeholder."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_sentences(path: str, column: int, sample_size: Optional[int]) -> List[str]:
    """Load sentences from a CSV file (optionally subsample)."""
    df = pd.read_csv(path)
    sentences = df.iloc[:, column].astype(str)

    if sample_size is not None:
        sentences = sentences.sample(n=sample_size, random_state=RANDOM_SEED)

    return sentences.tolist()


def build_prompts(template: str, sentences: List[str], use_chat: bool,
                  tokenizer: Optional[AutoTokenizer]) -> List[str]:
    """Return a list of formatted prompts ready for vLLM."""
    if use_chat and tokenizer is not None:
        messages = [[{"role": "user", "content": template.replace("{{SENTENCE}}", s)}]
                    for s in sentences]
        return [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
    # Fallback: plain string replacement (user supplies full template).
    return [template.replace("{{SENTENCE}}", s) for s in sentences]
#%%

def save_results_to_file(results: List[str], filename: str) -> None:
    """Save results to a JSONL file."""
    with open(filename, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Results written to {filename}")

def main() -> None:
    print("Loading prompt template …")
    prompt_template = load_prompt_template(PROMPT_FILE)

    print("Loading sentences …")
    sentences = load_sentences(DATA_FILE, SENTENCE_COLUMN, SAMPLE_SIZE)
    print(f"Total sentences to annotate: {len(sentences):,}")

    # Prepare tokenizer only if we intend to use chat formatting.
    tokenizer = None
    if USE_CHAT_TEMPLATE:
        print("Initialising tokenizer for chat template …")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    prompts = build_prompts(prompt_template, sentences, USE_CHAT_TEMPLATE, tokenizer)


    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
    )

    print("Initialising vLLM … (this may take a moment)")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILISATION,
    )

    results: List[str] = []

    # Get the len of the prompts list.

    num_prompts = len(prompts)
    print(f"Total prompts to process: {num_prompts:,}")
    print(f"Batch size: {BATCH_SIZE:,}")
    print(f"Save interval: Every {SAVE_INTERVAL:,} samples")

    print("Running inference …")
    for start in range(0, num_prompts, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_prompts)
        batch_prompts = prompts[start:end]
        print(f"Processing prompts {start:,} to {end:,} of {num_prompts:,}")
        # vLLM expects a list of strings (or list of lists for chat).
        outputs = llm.generate(batch_prompts, sampling_params)

        # RequestOutput -> best candidate text.
        results.extend([o.outputs[0].text.strip() for o in outputs])

        # Save intermediate results at specified intervals
        if end % SAVE_INTERVAL == 0 or end == len(prompts):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            interim_filename = f"llama3_8b_raw_{end}_{timestamp}.jsonl"
            save_results_to_file(results[:end], interim_filename)
            print(f"Saved interim results to {interim_filename}")

    # ── Display 5 samples of results ─────────────────────────────────────────────────────
    print("Displaying results …")
    for i in range(5):
        print(f"Response {i+1}: {results[i]}")
        print("-" * 80)
    # ───────────────────────────────────────────────────────────────────────────────
    
    # Save final results with the original filename
    save_results_to_file(results, "llama3_8b_raw.jsonl")
    
    print("Done.")   

if __name__ == "__main__":
    main()
# %%