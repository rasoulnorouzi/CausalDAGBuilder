import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def merge_overlapping_spans(spans: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """
    Merge overlapping spans that have the same label.
    Only merges spans with identical labels.
    """
    if not spans:
        return spans
    
    # Group spans by label
    spans_by_label = {}
    for span in spans:
        label = span['label']
        if label not in spans_by_label:
            spans_by_label[label] = []
        spans_by_label[label].append(span.copy())
    
    merged_spans = []
    
    # Process each label group separately
    for label, label_spans in spans_by_label.items():
        # Sort spans by start offset
        sorted_spans = sorted(label_spans, key=lambda x: x['start_offset'])
        
        # Initialize with first span
        current_span = sorted_spans[0].copy()
        
        # Try to merge with subsequent spans
        for next_span in sorted_spans[1:]:
            if current_span['end_offset'] >= next_span['start_offset']:
                # Merge overlapping spans
                current_span['end_offset'] = max(current_span['end_offset'], next_span['end_offset'])
            else:
                # No overlap, add current span and start new one
                merged_spans.append(current_span)
                current_span = next_span.copy()
        
        # Add the last span
        merged_spans.append(current_span)
    
    # Sort all spans by start offset for consistent output
    return sorted(merged_spans, key=lambda x: x['start_offset'])

def validate_span(span: Dict[str, Any], text: str) -> bool:
    """
    Validate that a span's offsets are valid and match the text.
    """
    try:
        if not all(k in span for k in ['start_offset', 'end_offset', 'label']):
            logger.debug(f"Span missing required fields: {span}")
            return False
            
        # Convert string offsets to integers if needed
        try:
            start_offset = int(span['start_offset'])
            end_offset = int(span['end_offset'])
            span['start_offset'] = start_offset
            span['end_offset'] = end_offset
        except (ValueError, TypeError):
            logger.debug(f"Invalid offset types in span: {span}")
            return False
            
        if start_offset < 0 or end_offset > len(text):
            logger.debug(f"Span offsets out of range: {span}, text length: {len(text)}")
            return False
            
        if start_offset >= end_offset:
            logger.debug(f"Invalid span offsets (start >= end): {span}")
            return False
            
        # Verify the span text matches
        span_text = text[start_offset:end_offset]
        if not span_text.strip():
            logger.debug(f"Empty span text: {span}")
            return False
            
        return True
    except Exception as e:
        logger.warning(f"Error validating span: {e}")
        return False

def normalize_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize relation types to match doccano format.
    """
    type_mapping = {
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
        'zero': 'zero',
        'pos': 'positive',
        'neg': 'negative',
        'neut': 'neutral',
        '0': 'zero',
        'Positive': 'positive',
        'Negative': 'negative',
        'Neutral': 'neutral',
        'Zero': 'zero'
    }
    
    normalized = []
    for rel in relations:
        try:
            rel_copy = rel.copy()
            rel_type = rel.get('type', '').lower()
            if not rel_type:
                rel_type = rel.get('polarity', '').lower()
            rel_copy['type'] = type_mapping.get(rel_type, 'neutral')
            normalized.append(rel_copy)
        except Exception as e:
            logger.warning(f"Error normalizing relation: {e}, relation: {rel}")
            continue
    
    return normalized

def find_span_in_text(text: str, span_text: str) -> Optional[Dict[str, Any]]:
    """
    Find the start and end offsets of a span in the text.
    Returns None if the span is not found.
    """
    if not span_text or not text:
        return None
        
    try:
        # Clean up the text and span
        text = text.strip()
        span_text = span_text.strip()
        
        start = text.find(span_text)
        if start == -1:
            # Try case-insensitive search
            text_lower = text.lower()
            span_text_lower = span_text.lower()
            start = text_lower.find(span_text_lower)
            if start == -1:
                return None
                
        return {
            'start_offset': start,
            'end_offset': start + len(span_text)
        }
    except Exception as e:
        logger.warning(f"Error finding span in text: {e}, span_text: {span_text}")
        return None

def clean_json_string(line: str) -> Dict[str, Any]:
    """
    Clean a JSON string and return a dictionary.
    Handles both nested JSON (with 'output' field) and direct JSON formats.
    """
    try:
        # First try to parse the outer JSON
        data = json.loads(line)
        
        # If it's a string, try to parse it as JSON
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                # If it's not valid JSON, wrap it in a minimal structure
                return {"text": data.strip()}
        
        # If there's an 'output' field that's a string, parse it as well
        if isinstance(data, dict):
            if 'output' in data and isinstance(data['output'], str):
                try:
                    output_data = json.loads(data['output'])
                    # Merge the output data with the original data
                    data.update(output_data)
                    # Remove the original output field
                    data.pop('output', None)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse output field as JSON: {data['output']}")
            return data
        else:
            # If data is not a dict, wrap it in a minimal structure
            return {"text": str(data)}
            
    except json.JSONDecodeError:
        # Remove any BOM characters
        line = line.strip().replace('\ufeff', '')
        
        # Handle escaped quotes
        if line.startswith('"') and line.endswith('"\n'):
            try:
                # Try to parse as escaped JSON string
                data = json.loads(line)
                if isinstance(data, str):
                    try:
                        return json.loads(data)
                    except json.JSONDecodeError:
                        return {"text": data.strip()}
                return data
            except json.JSONDecodeError:
                # Remove escape characters
                line = line.strip().strip('"').replace('\\"', '"')
        
        # Fix unescaped quotes within text fields
        line = line.replace('"s ', "'s ").replace('``', '"').replace("''", '"')
        
        # Try to fix common JSON formatting issues
        try:
            # Parse as Python literal (safer than eval)
            import ast
            data = ast.literal_eval(line)
            if isinstance(data, str):
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return {"text": data.strip()}
            return data
        except:
            try:
                # Last resort: try to fix quotes and parse
                import re
                # Fix quotes in text field
                text_pattern = r'"text":\s*"([^"]*)"'
                line = re.sub(text_pattern, lambda m: f'"text": "{m.group(1).replace(""", "\'").replace(""", "\'")}"', line)
                data = json.loads(line)
                return data
            except:
                # If all else fails, create a minimal valid JSON object
                return {"text": line.strip()}

def convert_llm_output_to_doccano(input_file: str, output_dir: str) -> str:
    """
    Convert LLM output file to doccano format.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save converted file
        
    Returns:
        Path to the converted file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    input_filename = os.path.basename(input_file)
    output_filename = f"{os.path.splitext(input_filename)[0]}_converted.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    
    converted_count = 0
    error_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            total_count += 1
            try:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Clean and parse the JSON
                data = clean_json_string(line.strip())
                
                # Handle different input formats
                text = data.get('text', '') or data.get('sentence', '') or data.get('content', '')
                if not text:
                    logger.warning(f"Line {line_num}: No text field found in data: {data}")
                    error_count += 1
                    continue
                
                # Extract entities/spans
                entities = []
                entity_id = 1
                entity_map = {}  # Map from text to entity ID
                
                # Try different possible locations for entity information
                spans_data = (data.get('entities', []) or 
                            data.get('spans', []) or 
                            data.get('annotations', []))
                
                # If no explicit spans, try to extract from relations
                if not spans_data and 'relations' in data:
                    for rel in data['relations']:
                        # Try to find cause and effect spans
                        for role, span_text in [('cause', rel.get('cause')), ('effect', rel.get('effect'))]:
                            if span_text and span_text not in entity_map:
                                span = find_span_in_text(text, span_text)
                                if span:
                                    entity = {
                                        'id': entity_id,
                                        'label': role,
                                        'start_offset': span['start_offset'],
                                        'end_offset': span['end_offset']
                                    }
                                    entities.append(entity)
                                    entity_map[span_text] = entity_id
                                    entity_id += 1
                
                # Process explicit spans if available
                for span in spans_data:
                    if validate_span(span, text):
                        entity = {
                            'id': entity_id,
                            'label': span['label'],
                            'start_offset': span['start_offset'],
                            'end_offset': span['end_offset']
                        }
                        entities.append(entity)
                        entity_id += 1
                
                # Merge overlapping spans with same label
                entities = merge_overlapping_spans(entities, text)
                
                # Extract and normalize relations
                relations = []
                if 'relations' in data:
                    for rel in data['relations']:
                        # Try to find cause and effect IDs
                        cause_text = rel.get('cause')
                        effect_text = rel.get('effect')
                        
                        if cause_text and effect_text:
                            # If we don't have the entity IDs yet, create them
                            if cause_text not in entity_map:
                                span = find_span_in_text(text, cause_text)
                                if span:
                                    entity = {
                                        'id': entity_id,
                                        'label': 'cause',
                                        'start_offset': span['start_offset'],
                                        'end_offset': span['end_offset']
                                    }
                                    entities.append(entity)
                                    entity_map[cause_text] = entity_id
                                    entity_id += 1
                            
                            if effect_text not in entity_map:
                                span = find_span_in_text(text, effect_text)
                                if span:
                                    entity = {
                                        'id': entity_id,
                                        'label': 'effect',
                                        'start_offset': span['start_offset'],
                                        'end_offset': span['end_offset']
                                    }
                                    entities.append(entity)
                                    entity_map[effect_text] = entity_id
                                    entity_id += 1
                            
                            # Now create the relation
                            if cause_text in entity_map and effect_text in entity_map:
                                relation = {
                                    'id': len(relations) + 1,
                                    'from_id': entity_map[cause_text],
                                    'to_id': entity_map[effect_text],
                                    'type': rel.get('polarity', 'neutral').lower()
                                }
                                relations.append(relation)
                
                relations = normalize_relations(relations)
                
                # Create doccano format entry
                doccano_entry = {
                    'id': data.get('id', line_num),
                    'text': text,
                    'entities': entities,
                    'relations': relations,
                    'Comments': []
                }
                
                # Write to output file
                f_out.write(json.dumps(doccano_entry) + '\n')
                converted_count += 1
                
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {str(e)}\n{traceback.format_exc()}")
                error_count += 1
                continue
    
    logger.info(f"Conversion complete: {converted_count} entries converted, {error_count} errors out of {total_count} total entries")
    logger.info(f"Output saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage
    input_files = [
        "datasets/llama3_8b_raw_5000_20250505_155238.jsonl",
        "datasets/llama38b_raw.jsonl",
        "datasets/llm33b_raw.jsonl",
        "datasets/qwen25_7b_raw.jsonl"
    ]
    
    output_dir = "datasets/converted"
    
    for input_file in input_files:
        try:
            output_path = convert_llm_output_to_doccano(input_file, output_dir)
            logger.info(f"Successfully converted {input_file}")
        except Exception as e:
            logger.error(f"Failed to convert {input_file}: {str(e)}\n{traceback.format_exc()}") 