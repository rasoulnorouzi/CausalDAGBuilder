import json
import spacy
import pandas as pd
from typing import List, Dict

class AnnotationProcessor:
    """
    A simple processor for converting entity annotations to token-level labels.
    
    Example:
        # Create sample input file (annotations.jsonl):
        {"text": "The rain caused flooding.", "entities": [{"label": "cause", "start_offset": 4, "end_offset": 8}]}
        
        # Use in Python:
        processor = AnnotationProcessor()
        df = processor.process_file('annotations.jsonl')
        print(df)
        
        # Save results if needed:
        df.to_csv('output.csv', index=False)
    """
    
    def __init__(self):
        """Initialize with spaCy's basic English tokenizer."""
        self.nlp = spacy.blank('en')

    def process_file(self, file_path: str) -> pd.DataFrame:
        """
        Process a JSONL annotation file and return token-level labels.
        
        Args:
            file_path: Path to JSONL file with annotations
            
        Returns:
            DataFrame with columns 'Token' and 'Labels'
        """
        # Load annotations
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        
        # Process annotations
        tokens_and_labels = []
        
        for record in data:
            text = record.get("text", "")
            entities = record.get("entities", [])
            
            # Tokenize text
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            labels = ["NONE"] * len(tokens)
            
            # Skip if no entities or all non-causal
            if not entities or all(entity.get("label") == "non-causal" for entity in entities):
                for token, label in zip(tokens, labels):
                    tokens_and_labels.append({"Token": token, "Labels": label})
                continue
            
            # Assign labels to tokens
            for entity in entities:
                label = entity.get("label")
                start = entity.get("start_offset")
                end = entity.get("end_offset")
                
                for i, token in enumerate(doc):
                    # Check if token is within entity span
                    if not (token.idx + len(token.text) <= start or token.idx >= end):
                        labels[i] = label if labels[i] == "NONE" else f"{labels[i]}+{label}"
            
            # Store tokens and labels
            for token, label in zip(tokens, labels):
                tokens_and_labels.append({"Token": token, "Labels": label})
        
        return pd.DataFrame(tokens_and_labels)