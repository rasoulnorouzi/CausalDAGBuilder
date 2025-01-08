import json
import spacy
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

class AnnotationProcessor:
    """
    Process annotation files to create token-level labels from entity annotations.
    
    This class processes JSONL annotation files containing text and entity annotations,
    converting them into token-level labels. It's particularly useful for processing
    outputs from annotation tools and preparing data for token classification tasks.
    
    Example JSONL input format:
    ```
    {"text": "The heavy rain caused flooding.", "entities": [{"label": "cause", "start_offset": 4, "end_offset": 14}]}
    {"text": "Due to the storm, trees fell.", "entities": [{"label": "effect", "start_offset": 16, "end_offset": 25}]}
    ```
    
    Basic usage:
    ```python
    # Initialize processor
    processor = AnnotationProcessor()
    
    # Process single file
    df = processor.process_file('annotations.jsonl')
    print(df)
    
    # Process multiple files
    results = processor.process_multiple_files(['ann1.jsonl', 'ann2.jsonl'])
    for annotator, df in results.items():
        print(f"{annotator}:")
        print(df)
        # Save to CSV
        df.to_csv(f'{annotator}_processed.csv', index=False)
    ```
    
    Output DataFrame format:
    ```
    Token    Labels
    The      NONE
    heavy    cause
    rain     cause
    caused   NONE
    flooding NONE
    ```
    
    Notes:
        - Tokens with no entity labels are marked as "NONE"
        - Overlapping entities are combined with "+" (e.g., "cause+effect")
        - Non-causal entities are treated as "NONE"
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the annotation processor.
        
        Args:
            spacy_model (str): Name of spaCy model to use for tokenization.
                              Default is "en_core_web_sm".
        
        Example:
        ```python
        # Default initialization
        processor = AnnotationProcessor()
        
        # With specific spaCy model
        processor = AnnotationProcessor(spacy_model="en_core_web_lg")
        ```
        """
        try:
            self.nlp = spacy.load(spacy_model)
            self.nlp.select_pipes(enable=['tokenizer'])
        except OSError:
            print(f"Could not load {spacy_model}, falling back to blank 'en' model")
            self.nlp = spacy.blank('en')

    def process_file(self, file_path: str) -> pd.DataFrame:
        """
        Process a single annotation file and return results as DataFrame.
        
        Args:
            file_path (str): Path to the JSONL annotation file
            
        Returns:
            pd.DataFrame: DataFrame with columns 'Token' and 'Labels'
        
        Example:
        ```python
        # Process single file
        df = processor.process_file('annotator1.jsonl')
        
        # View results
        print(df)
        
        # Save to CSV
        df.to_csv('processed_annotations.csv', index=False)
        
        # Basic analysis
        label_counts = df['Labels'].value_counts()
        print("\nLabel distribution:")
        print(label_counts)
        ```
        """
        annotation_data = self._load_jsonl(file_path)
        processed_data = self._process_annotations(annotation_data)
        return self.create_dataframe(processed_data)
    
    def process_multiple_files(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Process multiple annotation files and return results for each.
        
        Args:
            file_paths (List[str]): List of paths to JSONL annotation files
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping file names to their
                                   processed DataFrames
        
        Example:
        ```python
        # Process multiple files
        files = ['annotator1.jsonl', 'annotator2.jsonl', 'annotator3.jsonl']
        results = processor.process_multiple_files(files)
        
        # Analyze each annotator's results
        for annotator, df in results.items():
            print(f"\n{annotator} statistics:")
            print(f"Total tokens: {len(df)}")
            print("Label distribution:")
            print(df['Labels'].value_counts())
            
            # Save individual results
            df.to_csv(f'{annotator}_processed.csv', index=False)
        
        # Compare annotators
        for annotator, df in results.items():
            cause_count = len(df[df['Labels'] == 'cause'])
            effect_count = len(df[df['Labels'] == 'effect'])
            print(f"\n{annotator}:")
            print(f"Cause labels: {cause_count}")
            print(f"Effect labels: {effect_count}")
        ```
        """
        results = {}
        for file_path in file_paths:
            name = Path(file_path).stem
            results[name] = self.process_file(file_path)
        return results

    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """
        Load and parse JSONL file.
        
        Args:
            file_path (str): Path to JSONL file
            
        Returns:
            List[Dict]: List of parsed JSON objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                            f"Error parsing JSON at line {line_num}: {str(e)}",
                            doc=line,
                            pos=e.pos
                        )
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Annotation file not found: {file_path}")

    def _process_annotations(self, records: List[Dict]) -> List[Dict[str, Any]]:
        """
        Process annotation records into token-label pairs.
        
        Args:
            records (List[Dict]): List of annotation records
            
        Returns:
            List[Dict[str, Any]]: List of processed records with tokens and labels
        """
        token_label_data = []
        
        for record in records:
            text = record.get("text", "")
            entities = record.get("entities", [])
            
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            token_labels = ["NONE"] * len(tokens)
            
            if not entities or all(entity.get("label") == "non-causal" for entity in entities):
                token_label_data.append({"tokens": tokens, "labels": token_labels})
                continue
            
            for entity in entities:
                label = entity.get("label")
                start_offset = entity.get("start_offset")
                end_offset = entity.get("end_offset")
                
                for i, token in enumerate(doc):
                    if not (token.idx + len(token.text) <= start_offset or token.idx >= end_offset):
                        if token_labels[i] == "NONE":
                            token_labels[i] = label
                        else:
                            token_labels[i] += f"+{label}"
            
            token_label_data.append({"tokens": tokens, "labels": token_labels})
        
        return token_label_data

    def create_dataframe(self, annotation_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert processed annotation data to DataFrame.
        
        Args:
            annotation_data (List[Dict[str, Any]]): Processed annotation data
            
        Returns:
            pd.DataFrame: DataFrame with 'Token' and 'Labels' columns
            
        Example:
        ```python
        # Convert processed data to DataFrame
        df = processor.create_dataframe(processed_data)
        
        # Basic DataFrame operations
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nLabel distribution:")
        print(df['Labels'].value_counts())
        
        print("\nUnique tokens:")
        print(df['Token'].nunique())
        ```
        """
        data = []
        for record in annotation_data:
            for token, label in zip(record["tokens"], record["labels"]):
                data.append({"Token": token, "Labels": label})
        return pd.DataFrame(data)