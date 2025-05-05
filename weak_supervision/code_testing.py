# %%
import json
from llm2doccano import convert_llm_output_to_doccano_format
# %%
# llama38b_raw.jsonl
with open("/home/rnorouzini/CausalDAGBuilder/weak_supervision/llama3_8b_raw.jsonl", "r", encoding="utf-8") as f:
    # Each line is a JSON string, so we need to parse it once

    # raw_lines = f.readlines()
    # llama38b_data = []
    # for line in raw_lines:
    #     # Parse the JSON string to get the actual data
    #     parsed_json = json.loads(line)  
    #     # Format the data as expected by the converter function
    #     # The function expects objects with an "output" field
    #     llama38b_data.append({"output": parsed_json})

    readed_llama38b_data = [json.loads(line) for line in f.readlines()]

# %%
readed_llama38b_data[0]
# %%
