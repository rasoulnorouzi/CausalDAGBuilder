# %%
import json
from llm2doccano import convert_llm_output_to_doccano_format
# %%
# llama38b_raw.jsonl
with open("/home/rnorouzini/CausalDAGBuilder/weak_supervision/llama3_8b_raw.jsonl", "r", encoding="utf-8") as f:
    # Each line is a JSON string, so we need to parse it once

    raw_lines = f.readlines()
    llama38b_data = []
    for line in raw_lines:
        # Parse the JSON string to get the actual data
        parsed_json = json.loads(line)  
        # Format the data as expected by the converter function
        # The function expects objects with an "output" field
        llama38b_data.append({"output": parsed_json})
    # Convert the data to the format expected by doccano
    doccano_data = convert_llm_output_to_doccano_format(llama38b_data)
    # Save the converted data to a new JSONL file
    with open("/home/rnorouzini/CausalDAGBuilder/weak_supervision/llama3_8b_doccano.jsonl", "w", encoding="utf-8") as f:
        for item in doccano_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Results written to llama3_8b_doccano.jsonl")
    print("Done.")
# %%
test = "the other two studies which had a single communication period allowed free oral communication, and this could have been more effective in producing higher cooperation than opportunities for standard message exchange."

test[172:215]
# %%
