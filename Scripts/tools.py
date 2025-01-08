# Re-import necessary libraries and reload paths after reset
import json
import spacy
import pandas as pd

# Reload the spaCy tokenizer
nlp = spacy.blank("en")

# File paths for the JSONL annotation files
rasoul_path = '/content/rasoul.jsonl'
caspar_path = '/content/caspar.jsonl'
bennett_path = '/content/bennett.jsonl'

def process_annotations_compact(file_path):
    """
    Process annotation data to represent multiple labels compactly for the same token.
    Tokens are generated using spaCy's tokenizer.
    If a sentence has no entities or is fully labeled as 'non-causal', all tokens get 'NONE'.

    Args:
        file_path (str): Path to the annotation file.

    Returns:
        list: A list of dictionaries, each containing tokens and their compact labels.
    """
    token_label_data = []

    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line)
            sentence = record.get("text", "")
            entities = record.get("entities", [])

            # Tokenize sentence using spaCy
            doc = nlp(sentence)
            tokens = [token.text for token in doc]
            token_labels = ["NONE"] * len(tokens)  # Default all tokens to NONE

            if not entities:
                # If no entities, return all tokens labeled as NONE
                token_label_data.append({"tokens": tokens, "labels": token_labels})
                continue

            # Check if all entities are "non-causal"
            all_non_causal = all(entity.get("label") == "non-causal" for entity in entities)

            if all_non_causal:
                # Skip processing and assign NONE to all tokens
                token_label_data.append({"tokens": tokens, "labels": token_labels})
                continue

            # Process each token and assign labels
            for entity in entities:
                label = entity.get("label")
                start_offset = entity.get("start_offset")
                end_offset = entity.get("end_offset")

                # Match tokens with the annotation spans
                for i, token in enumerate(doc):
                    if not (token.idx + len(token.text) <= start_offset or token.idx >= end_offset):
                        if token_labels[i] == "NONE":
                            token_labels[i] = label
                        else:
                            # Combine multiple labels compactly
                            token_labels[i] += f"+{label}"

            # Append token-label pairs
            token_label_data.append({"tokens": tokens, "labels": token_labels})

    return token_label_data



def create_compact_dataframe(annotation_data):
    """
    Convert compact annotation data into a dataframe for visualization.

    Args:
        annotation_data (list): List of dictionaries containing tokens and labels.

    Returns:
        pd.DataFrame: Dataframe with tokens and their compact labels.
    """
    data = []
    for record in annotation_data:
        for token, label in zip(record["tokens"], record["labels"]):
            data.append({"Token": token, "Labels": label})
    return pd.DataFrame(data)


# Process compact annotations for both annotators
rasoul_compact_data = process_annotations_compact(rasoul_path)
caspar_compact_data = process_annotations_compact(caspar_path)
bennett_compact_data = process_annotations_compact(bennett_path)

# Convert to dataframes for visualization
rasoul_compact_df = create_compact_dataframe(rasoul_compact_data)
caspar_compact_df = create_compact_dataframe(caspar_compact_data)
bennett_compact_df = create_compact_dataframe(bennett_compact_data)