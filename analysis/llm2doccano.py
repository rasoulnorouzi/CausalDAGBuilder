import difflib
import json
import re

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

def safe_json_loads(s):
    """
    Attempts to fix common JSON escape issues before parsing.
    Returns None if parsing still fails.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        # Try to escape backslashes that are not part of valid escape sequences
        import re
        # Replace single backslashes not followed by ["\\/bfnrtu] with double backslash
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
        try:
            return json.loads(fixed)
        except Exception as e2:
            print(f"Failed to fix JSON: {e2}")
            return None
    except Exception as e:
        print(f"Other JSON error: {e}")
        return None

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
        sentence_data = safe_json_loads(item["output"])
        if sentence_data is None:
            print(f"Skipping sample {i+1}: still invalid JSON in 'output' field.")
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