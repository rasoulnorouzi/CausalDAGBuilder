# %%
import os
import re
import glob
import logging
import fitz
import ftfy
from langdetect import detect, DetectorFactory
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import wordninja
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import pandas as pd
import time
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset
# %%


# Precompile regex patterns
PATTERN_WHITESPACE = re.compile(r'\s+')
PATTERN_LEADING_NONLETTER = re.compile(r'^[^A-Za-z]+')
PATTERN_UNMATCHED_PARENS = re.compile(r'\([^)]*$')
PATTERN_HYPHEN_SPLIT = re.compile(r'(\w+)-\s+(\w+)')


# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Global variables for model, tokenizer, and device
tokenizer = None
model = None
device = None

def init_worker():
    """
    Initialize each worker process:
      - Set the device (GPU if available, otherwise CPU)
      - Load the transformer tokenizer and model
    """
    global tokenizer, model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Worker process using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("rasoultilburg/ssc_bert")
    model = AutoModelForSequenceClassification.from_pretrained("rasoultilburg/ssc_bert")
    model.to(device)
    model.eval()  # Set the model to evaluation mode



# Extract text from a PDF file using PyMuPDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Error opening {pdf_path}: {e}")
        return None
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    
    # Normalize the text using ftfy
    text = ftfy.fix_text(text)

    # Lowercase the text
    text = text.lower()

    # Remove extra whitespace
    text = PATTERN_WHITESPACE.sub(' ', text)

    # Fix hyphenated words
    text = PATTERN_HYPHEN_SPLIT.sub(r'\1\2', text)

    return text

def is_english(text, seed=8642):
    """Check if the given text is in English."""
    DetectorFactory.seed = seed
    try:
        lang = detect(text)
    except Exception as e:
        logging.error(f"Error detecting language: {e}")
        return False
    return lang == 'en'

def clean_sentence(sentence):
    """
    Clean and standardize a sentence:
      - Remove leading non-letter characters
      - Replace fancy quotes and dashes with standard ones
      - Trim extra whitespace
    """
    # Ensure it starts with a letter
    sentence = PATTERN_LEADING_NONLETTER.sub('', sentence)

    # Replace fancy quotes/dashes and other characters
    sentence = (sentence.replace('‘', "'")
                        .replace('’', "'")
                        .replace('“', '"')
                        .replace('”', '"')
                        .replace('–', '-')
                        .replace('—', '-')
                        .replace('…', '...'))

    # Trim any leftover whitespace
    sentence = sentence.strip()

    return sentence


def split_concatenated_words(sentence, threshold=15):
    """Use wordninja to split concatenated words in the sentence."""
    words = sentence.split()
    split_words = []
    for word in words:
        if len(word) > threshold:
            split_words.extend(wordninja.split(word))
        else:
            split_words.append(word)
    return ' '.join(split_words)


def extract_sentences(text, threshold_s=10, threshold_e=50):
    """
    Extract sentences from the given text:
      - Split the text into sentences
      - Keep sentences within the length thresholds with split words
      - Clean and standardize each sentence

    Returns a list of cleaned sentences.
    """
    sentences = sent_tokenize(text)
    filtered_sentences = [sentence for sentence in sentences if threshold_s <= len(sentence.split()) <= threshold_e]
    cleaned_sentences = [clean_sentence(sentence) for sentence in filtered_sentences]
    cleaned_sentences = [split_concatenated_words(sentence) for sentence in cleaned_sentences]
    
    return cleaned_sentences


# PDF files to Senteces df
def pdf_to_sentences(pdf_path):
    """
    Extract sentences from a PDF file:
      - Extract text from the PDF
      - Check if the text is in English
      - Clean the text
      - Extract sentences from the cleaned text
    """
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None
        
    # Check if text is in English
    if not is_english(text):
        logging.warning(f"Skipping non-English file: {pdf_path}")
        return None
        
    cleaned_text = clean_text(text)
    sentences = extract_sentences(cleaned_text)
    return sentences

def pdf_dir_to_sentences(pdf_dir):
    """
    Extract sentences from all English PDF files in a directory with clean progress reporting.
    """
    pdf_files = glob.glob(os.path.join(pdf_dir, '*.pdf'))
    all_sentences = []
    all_pdf_files = []
    skipped_files = []
    
    total_files = len(pdf_files)
    print("\nProcessing PDF files...")
    print("-" * 50)
    
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1  # Update progress every second
    total_sentences = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        # Process current PDF
        current_sentences = pdf_to_sentences(pdf_file)
        
        # Update counts and lists
        if current_sentences:
            sentences_in_file = len(current_sentences)
            all_sentences.extend(current_sentences)
            all_pdf_files.extend([pdf_file] * sentences_in_file)
            total_sentences += sentences_in_file
        else:
            skipped_files.append(pdf_file)
            
        # Update progress at interval
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            elapsed_time = current_time - start_time
            progress_percent = (i / total_files) * 100
            files_per_second = i / elapsed_time if elapsed_time > 0 else 0
            eta = (elapsed_time / progress_percent) * (100 - progress_percent) if progress_percent > 0 else 0
            
            print(f"\rProgress: {progress_percent:>5.1f}% | "
                  f"Files: {i:>4}/{total_files:<4} | "
                  f"Sentences: {total_sentences:>6} | "
                  f"Speed: {files_per_second:>4.1f} files/s | "
                  f"ETA: {eta:>4.0f}s", end='')
            
            last_update_time = current_time
    
    # Final statistics
    total_time = time.time() - start_time
    successful_files = total_files - len(skipped_files)
    
    print("\n" + "-" * 50)
    print("PDF Processing complete!")
    print(f"Time: {total_time:.1f}s | "
          f"Processed: {successful_files}/{total_files} files | "
          f"Extracted: {len(all_sentences)} sentences "
          f"({len(all_sentences)/successful_files:.1f} per file)")
    
    return all_sentences, all_pdf_files


# Convert sentences, all_pdf_files to dataset
def sentences_to_dataset(sentences, pdf_files):
    """
    Convert a list of sentences and corresponding PDF files to a Hugging Face Dataset.
    """
    dataset = Dataset.from_dict({
        'pdf_file': pdf_files,
        'sentence': sentences
    })
    return dataset



# Classify sentences

def classify_sentences(pdf_sentences_dataset, batch_size=16):
    """
    Classify sentences using HuggingFace's optimized data processing pipeline.
    Returns:
        tuple: (causal_sentences, causal_pdf_files) containing:
            - list of sentences classified as causal
            - list of corresponding PDF files for each causal sentence
    """
    global model, device
    
    total_sentences = len(pdf_sentences_dataset)
    print(f"\nStarting sentence classification for {total_sentences} sentences...")
    
    # Store original sentences and PDF files
    original_sentences = pdf_sentences_dataset['sentence']
    original_pdf_files = pdf_sentences_dataset['pdf_file']
    
    # Get unique PDF files for progress tracking
    unique_pdfs = len(set(original_pdf_files))
    print(f"Processing sentences from {unique_pdfs} unique PDF files")
    
    print("Tokenizing sentences...")
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], truncation=True)
    
    tokenized_dataset = pdf_sentences_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=pdf_sentences_dataset.column_names
    )
    
    print("Creating DataLoader...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    predictions = []
    total_batches = len(dataloader)
    current_causal_count = 0
    start_time = time.time()
    
    print("\nClassifying sentences...")
    print(f"Total batches to process: {total_batches} (batch size: {batch_size})")
    
    for batch_idx, batch in enumerate(dataloader, 1):
        try:
            # Process batch
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).tolist()
            predictions.extend(batch_preds)
            
            # Update causal count
            current_causal_count += sum(1 for pred in batch_preds if pred == 1)
            
            # Calculate progress statistics
            sentences_processed = min(batch_idx * batch_size, total_sentences)
            progress_percent = (sentences_processed / total_sentences) * 100
            elapsed_time = time.time() - start_time
            sentences_per_second = sentences_processed / elapsed_time if elapsed_time > 0 else 0
            
            # Print progress update
            print(f"\rProgress: {progress_percent:.1f}% | "
                  f"Batch: {batch_idx}/{total_batches} | "
                  f"Sentences: {sentences_processed}/{total_sentences} | "
                  f"Found Causal: {current_causal_count} | "
                  f"Speed: {sentences_per_second:.1f} sentences/sec", end='')
            
            # Print detailed stats every 50 batches
            if batch_idx % 50 == 0:
                print(f"\nDetailed Stats at batch {batch_idx}:")
                print(f"- Causal sentences found so far: {current_causal_count}")
                print(f"- Causal ratio: {(current_causal_count/sentences_processed)*100:.1f}%")
                print(f"- Time elapsed: {elapsed_time:.1f} seconds")
                print(f"- Estimated time remaining: {(elapsed_time/progress_percent)*100 - elapsed_time:.1f} seconds")
                print("-" * 50)
            
        except Exception as e:
            logging.error(f"Error during batch {batch_idx}: {e}")
            predictions.extend([0] * len(batch))
    
    print("\n\nClassification complete!")
    
    # Filter causal sentences
    causal_sentences = []
    causal_pdf_files = []
    
    for pred, sent, pdf in zip(predictions, original_sentences, original_pdf_files):
        if pred == 1:
            causal_sentences.append(sent)
            causal_pdf_files.append(pdf)
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\nClassification Statistics:")
    print(f"- Total processing time: {total_time:.2f} seconds")
    print(f"- Average speed: {total_sentences/total_time:.1f} sentences/second")
    print(f"- Total sentences processed: {total_sentences}")
    print(f"- Total causal sentences found: {len(causal_sentences)}")
    print(f"- Overall causal ratio: {(len(causal_sentences)/total_sentences)*100:.1f}%")
    
    return causal_sentences, causal_pdf_files


# No it's time to put all the pieces together and process a directory of PDF files, extracting causal sentences from each one.The function name has causal in it!

def get_causal_sentences(pdf_dir, batch_size=16):
    """
    Process a directory of PDF files:
      - Extract sentences from each PDF
      - Classify sentences as causal or non-causal
    Returns:
        tuple: (causal_sentences, causal_pdf_files) containing:
            - list of sentences classified as causal
            - list of corresponding PDF files for each causal sentence
    """

    print("\n=== Starting Causal Sentence Extraction Pipeline ===")
    start_time = time.time()
    
    init_worker()
    
    print("\nStep 1: Extracting sentences from PDF files...")
    sentences, pdf_files = pdf_dir_to_sentences(pdf_dir)
    
    print("\nStep 2: Creating dataset...")
    pdf_sentences_dataset = sentences_to_dataset(sentences, pdf_files)
    
    print("\nStep 3: Classifying sentences...")
    causal_sentences, causal_pdf_files = classify_sentences(pdf_sentences_dataset, batch_size=batch_size)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\n=== Pipeline Complete ===")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Total sentences processed: {len(sentences)}")
    print(f"Total causal sentences found: {len(causal_sentences)}")
    print("================================")
    
    return causal_sentences, causal_pdf_files

# lets test the function with a sample directory of PDF files

# Sample PDF directory
pdf_directory = "data"
causal_sentences, causal_pdf_files = get_causal_sentences(pdf_directory, batch_size=16)

# Export the results to a CSV file
results_df = pd.DataFrame({
    'pdf_file': causal_pdf_files,
    'causal_sentence': causal_sentences
})

output_csv = "causal_sentences.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nResults saved to: {output_csv}")
