import json
import spacy
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

class AnnotationProcessor:
    """Process annotation files and create token-level labels."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the processor.
        
        Args:
            spacy_model: Name of spaCy model to use (default: "en_core_web_sm")
        """
        try:
            self.nlp = spacy.load(spacy_model)
            # Disable unnecessary components for better performance
            self.nlp.select_pipes(enable=['tokenizer'])
        except OSError:
            print(f"Could not load {spacy_model}, falling back to blank 'en' model")
            self.nlp = spacy.blank('en')

    def process_file(self, file_path: str) -> pd.DataFrame:
        """
        Process a single annotation file and return results as DataFrame.
        
        Args:
            file_path: Path to the JSONL annotation file
            
        Returns:
            DataFrame with tokens and their labels
        """
        annotation_data = self._load_jsonl(file_path)
        processed_data = self._process_annotations(annotation_data)
        return self.create_dataframe(processed_data)
    
    def process_multiple_files(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Process multiple annotation files.
        
        Args:
            file_paths: List of paths to JSONL annotation files
            
        Returns:
            Dictionary mapping file names to their processed DataFrames
        """
        results = {}
        for file_path in file_paths:
            name = Path(file_path).stem
            results[name] = self.process_file(file_path)
        return results

    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """Load and parse JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def _process_annotations(self, records: List[Dict]) -> List[Dict[str, Any]]:
        """Process annotation records into token-label pairs."""
        token_label_data = []
        
        for record in records:
            text = record.get("text", "")
            entities = record.get("entities", [])
            
            # Tokenize sentence using spaCy
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            token_labels = ["NONE"] * len(tokens)  # Default all tokens to NONE
            
            if not entities or all(entity.get("label") == "non-causal" for entity in entities):
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
                            token_labels[i] += f"+{label}"
            
            token_label_data.append({"tokens": tokens, "labels": token_labels})
        
        return token_label_data

    def create_dataframe(self, annotation_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert processed annotation data to DataFrame."""
        data = []
        for record in annotation_data:
            for token, label in zip(record["tokens"], record["labels"]):
                data.append({"Token": token, "Labels": label})
        return pd.DataFrame(data)