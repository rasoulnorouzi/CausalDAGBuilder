import difflib
import json
import re

"""
llsm outputs formatted as:
{
  "text": "[exact input sentence]",
  "causal": true,
  "relations": [
     {
       "cause": "[exact cause text 1]",
       "effect": "[exact effect text 1]",
       "polarity": "[Polarity1]"
     },
     {
       "cause": "[exact cause text 2]",
       "effect": "[exact effect text 2]",
       "polarity": "[Polarity2]"
     }
     // ... potentially more relations
  ]
}
"""

def locate_best_matching_span(text, phrase, threshold=0.8):
    """
    Attempts to find the best fuzzy match for a given phrase within a larger text.
    This is useful when phrases provided by LLMs don't exactly match the sentence structure
    due to lemmatization, paraphrasing, or slight rewording.

    Parameters:
        text (str): The full sentence in which the phrase should appear.
        phrase (str): The causal or effect phrase to locate.
        threshold (float): Minimum similarity score for accepting a fuzzy match.

    Returns:
        tuple[int, int] | None: Start and end character indices of the matched span, or None if not found.
    """
    phrase_len = len(phrase)
    best_match = None
    highest_ratio = 0

    for i in range(len(text) - phrase_len + 1):
        window = text[i:i + phrase_len + 20]
        ratio = difflib.SequenceMatcher(None, phrase.lower(), window.lower()).ratio()
        if ratio > highest_ratio and ratio >= threshold:
            highest_ratio = ratio
            best_match = (i, i + len(window))

    return best_match

def clean_json_code_block(s):
    """
    Removes Markdown code block markers (e.g., ```json ... ```) and trims whitespace.
    """
    if not isinstance(s, str):
        return s
    s = s.strip()
    # Remove triple backticks and optional language specifier at the start
    if s.startswith("```"):
        lines = s.splitlines()
        # Remove first line if it starts with ```
        if lines and lines[0].startswith("```"):
            lines = lines[1:]  # Fixed: Was incorrectly nested as [lines[1:]]
        # Remove last line if it is ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Remove leading/trailing single backticks (rare)
    s = re.sub(r"^`+|`+$", "", s).strip()
    return s

def safe_json_loads(s):
    """
    Attempts to fix common JSON formatting issues from LLM outputs before parsing.
    Always returns a dict with at least 'text', 'relations', and 'causal' keys.
    """
    if not isinstance(s, str):
        print(f"safe_json_loads: Input is not a string: {type(s)}")
        return {"text": "", "relations": [], "causal": None}

    original_string_for_debugging = s

    # --- Step 1: Clean Markdown code blocks and outer whitespace ---
    s = clean_json_code_block(s)
    if not s:
        print("safe_json_loads: String is empty after cleaning code blocks.")
        return {"text": "", "relations": [], "causal": None}

    # --- Step 2: Remove problematic control characters (except \t, \n, \r) ---
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    
    # --- Special Case: Check if input is completely malformed and try to repair structure ---
    if not s.strip().startswith('{') and not s.strip().startswith('['):
        if re.search(r'"text"\s*:', s):
            s = '{' + s
        else:
            if ':' in s and ('"' in s or "'" in s):
                s = '{' + s + '}'
    if not s.strip().endswith('}') and not s.strip().endswith(']'):
        if s.strip().startswith('{'):
            s = s + '}'
        elif s.strip().startswith('['):
            s = s + ']'

    max_attempts = 10
    last_error = None

    for attempt in range(max_attempts):
        try:
            obj = json.loads(s)
            # Ensure template keys exist
            if "text" not in obj:
                obj["text"] = ""
            if "relations" not in obj:
                obj["relations"] = []
            if "causal" not in obj:
                obj["causal"] = None
            return obj
        except json.JSONDecodeError as e:
            last_error = e
            applied_fix = False
            error_msg = str(e)
            # ...existing fix steps...
            # Fix for "Expecting property name enclosed in double quotes"
            if "Expecting property name enclosed in double quotes" in error_msg:
                s_new = re.sub(r'([{,]\s*)([a-zA-Z0-9_\-]+)(\s*:)', r'\1"\2"\3', s)
                if s_new != s:
                    s = s_new
                    applied_fix = True
                    continue
            if "Expecting ',' delimiter" in error_msg or "Expecting ':' delimiter" in error_msg:
                open_braces = s.count('{')
                close_braces = s.count('}')
                open_brackets = s.count('[')
                close_brackets = s.count(']')
                if open_braces > close_braces:
                    s = s + ('}' * (open_braces - close_braces))
                    applied_fix = True
                if open_brackets > close_braces:
                    s = s + (']' * (open_brackets - close_braces))
                    applied_fix = True
                if applied_fix:
                    continue
            s_new = s.replace('tban', 'than').replace('tbe', 'the').replace('cboosing', 'choosing')
            s_new = s_new.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
            if s_new != s:
                s = s_new
                applied_fix = True
                continue
            s_new = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
            if s_new != s:
                s = s_new
                applied_fix = True
                continue
            s_new = re.sub(r'\\u([^0-9a-fA-F]|$|[0-9a-fA-F]{0,3}$)', r'\\\\u\1', s)
            if s_new != s:
                s = s_new
                applied_fix = True
                continue
            s_new = re.sub(r'(?<!\\)\n', r'\\n', s)
            if s_new != s:
                s = s_new
                applied_fix = True
                continue
            # ...existing code for quote fixes, trailing commas, etc...
            # Fix 3n: Remove whitespace and newlines right after '{' if it's followed by '"'
            s_new = re.sub(r'(\{)\s+(")', r'\1\2', s)
            if s_new != s:
                s = s_new
                applied_fix = True
                continue
            if not applied_fix:
                break

    # --- Last Resort: Extract fields manually to match template ---
    text_val = ""
    relations_val = []
    causal_val = None

    # Try to extract text
    text_match = re.search(r'"text"\s*:\s*"([^"]*)"', original_string_for_debugging)
    if text_match:
        text_val = text_match.group(1)

    # Try to extract causal
    causal_match = re.search(r'"causal"\s*:\s*(true|false|null)', original_string_for_debugging, re.IGNORECASE)
    if causal_match:
        val = causal_match.group(1).lower()
        if val == "true":
            causal_val = True
        elif val == "false":
            causal_val = False
        else:
            causal_val = None

    # Try to extract relations as a list of dicts
    relations_match = re.search(r'"relations"\s*:\s*(\[[^\]]*\])', original_string_for_debugging, re.DOTALL)
    if relations_match:
        try:
            rels = json.loads(relations_match.group(1))
            # Ensure each relation has the required keys
            relations_val = [
                {
                    "cause": rel.get("cause", ""),
                    "effect": rel.get("effect", ""),
                    "polarity": rel.get("polarity", "")
                }
                for rel in rels if isinstance(rel, dict)
            ]
        except Exception:
            relations_val = []
    # If not found, try to extract at least one cause/effect/polarity
    if not relations_val:
        rels = []
        rel_pattern = re.compile(
            r'\{\s*"cause"\s*:\s*"([^"]*)"\s*,\s*"effect"\s*:\s*"([^"]*)"\s*,\s*"polarity"\s*:\s*"([^"]*)"\s*\}',
            re.DOTALL)
        for m in rel_pattern.finditer(original_string_for_debugging):
            rels.append({"cause": m.group(1), "effect": m.group(2), "polarity": m.group(3)})
        relations_val = rels

    # Always return the template, even if all fields are empty
    return {"text": text_val, "relations": relations_val, "causal": causal_val}

def convert_llm_output_to_doccano_format(ollama_data):
    """
    Converts LLM-generated causal annotations to Doccano JSON format.

    This function reads LLM outputs (containing sentences and structured causal relationships)
    and transforms them into a format compatible with Doccano, including:
    - Generating token span offsets for each cause and effect phrase using exact and fuzzy matching
    - Assigning unique IDs to entities and relations
    - Normalizing polarity values
    - Structuring output as expected by Doccano with keys: id, text, entities, relations, Comments

    Parameters:
        ollama_data (list[dict]): A list of LLM annotation outputs, each with "sentence" and "output" fields.

    Returns:
        list[dict]: Doccano-compatible formatted data.
    """
    doccano_formatted = []
    global_entity_id = 1000
    global_relation_id = 500

    for i, item in enumerate(ollama_data):
        output_field = item.get("output", "")
        if not output_field or not isinstance(output_field, str):
            print(f"Sample {i+1} has missing or empty 'output' field: {output_field!r}")
            # Instead of continue, add a minimal record with empty text
            base = {
                "id": i+1,
                "text": "",
                "entities": [],
                "relations": [],
                "Comments": ["Invalid or missing output field"]
            }
            doccano_formatted.append(base)
            continue
        # Clean up possible code block or stray backticks
        output_field = clean_json_code_block(output_field)
        sentence_data = safe_json_loads(output_field)
        if sentence_data is None:
            # Try to extract text field manually
            text_match = re.search(r'"text"\s*:\s*"([^"]*)"', output_field)
            text_val = text_match.group(1) if text_match else ""
            base = {
                "id": i+1,
                "text": text_val,
                "entities": [],
                "relations": [],
                "Comments": ["Could not parse output field as JSON"]
            }
            doccano_formatted.append(base)
            continue

        base = {
            "id": i+1,
            "text": sentence_data["text"],
            "entities": [],
            "relations": [],
            "Comments": []
        }

        text = sentence_data["text"]
        entity_map = {}

        for relation in sentence_data.get("relations", []):
            for role in ["cause", "effect"]:
                phrase = relation[role]
                if phrase in entity_map:
                    continue

                match = re.search(re.escape(phrase), text)
                if match:
                    start_offset, end_offset = match.start(), match.end()
                else:
                    span = locate_best_matching_span(text, phrase)
                    if span:
                        start_offset, end_offset = span
                    else:
                        continue  # Skip if no match found

                entity_id = global_entity_id
                global_entity_id += 1
                entity_map[phrase] = entity_id

                base["entities"].append({
                    "id": entity_id,
                    "label": role,
                    "start_offset": start_offset,
                    "end_offset": end_offset
                })

            cause_id = entity_map.get(relation["cause"])
            effect_id = entity_map.get(relation["effect"])
            if cause_id is not None and effect_id is not None:
                base["relations"].append({
                    "id": global_relation_id,
                    "from_id": cause_id,
                    "to_id": effect_id,
                    "type": relation["polarity"].lower()
                })
                global_relation_id += 1

        doccano_formatted.append(base)

    return doccano_formatted


"""
How to use: 

# llm33b_raw.jsonl
with open("llm33b_raw.jsonl", "r", encoding="utf-8") as f:
    llm33b_data = [json.loads(line) for line in f.readlines()]
doccano_llm33b = convert_llm_output_to_doccano_format(llm33b_data)

# Save the Doccano formatted data to a JSON file
with open("annotation_datasets/doccano_llm33b.jsonl", "w", encoding="utf-8") as f:
    for sample in doccano_llm33b:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# llama38b_raw.jsonl
with open("llama38b_raw.jsonl", "r", encoding="utf-8") as f:
    llama38b_data = [json.loads(line) for line in f.readlines()]    

doccano_llama38b = convert_llm_output_to_doccano_format(llama38b_data)

# Save the Doccano formatted data to a JSON file
with open("annotation_datasets/doccano_llama38b.jsonl", "w", encoding="utf-8") as f:
    for sample in doccano_llama38b:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# gemma34b_raw.jsonl
with open("gemma34b_raw.jsonl", "r", encoding="utf-8") as f:
    gemma34b_data = [json.loads(line) for line in f.readlines()]

doccano_gemma34b = convert_llm_output_to_doccano_format(gemma34b_data)

# Save the Doccano formatted data to a JSON file
with open("annotation_datasets/doccano_gemma34b.jsonl", "w", encoding="utf-8") as f:
    for sample in doccano_gemma34b:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# gemma312b_raw.jsonl
with open("gemma312b_raw.jsonl", "r", encoding="utf-8") as f:
    gemma312b_data = [json.loads(line) for line in f.readlines()]

doccano_gemma312b = convert_llm_output_to_doccano_format(gemma312b_data)

# Save the Doccano formatted data to a JSON file
with open("annotation_datasets/doccano_gemma312b.jsonl", "w", encoding="utf-8") as f:
    for sample in doccano_gemma312b:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

"""