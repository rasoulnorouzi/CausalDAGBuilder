import json
import spacy
import itertools
from collections import Counter

# -------------------------------
# Annotation Processing Component
# -------------------------------

class AnnotationProcessor:
    def __init__(self, file_path, language="en"):
        """
        Initialize the processor with the given file path and spaCy language.
        """
        self.file_path = file_path
        spacy.prefer_gpu()
        self.nlp = spacy.blank(language)
        self.annotations = self._process_annotations()

    def _process_annotations(self):
        """
        Processes the annotation JSONL file into a list of dictionaries.
        Each dictionary contains:
          - 'tokens': list of tokenized words in the sentence.
          - 'labels': list of corresponding labels for each token.
        If a sentence has no entities or only "non-causal" entities, all tokens are labeled as "NONE".
        """
        processed = []
        with open(self.file_path, "r") as file:
            for line in file:
                record = json.loads(line)
                sentence = record.get("text", "")
                entities = record.get("entities", [])
                doc = self.nlp(sentence)
                tokens = [token.text for token in doc]
                labels = ["NONE"] * len(tokens)

                # If no entities or all are "non-causal", assign "NONE" to all tokens.
                if not entities or all(entity.get("label") == "non-causal" for entity in entities):
                    processed.append({"tokens": tokens, "labels": labels})
                    continue

                # Otherwise, assign labels based on entity spans.
                for entity in entities:
                    label = entity.get("label")
                    start_offset = entity.get("start_offset")
                    end_offset = entity.get("end_offset")
                    for i, token in enumerate(doc):
                        # Check if token overlaps with the entity span.
                        if not (token.idx + len(token.text) <= start_offset or token.idx >= end_offset):
                            if labels[i] == "NONE":
                                labels[i] = label
                            else:
                                # Split the current composite label into individual parts.
                                current_labels = labels[i].split("+")
                                # Add the new label if it's not already present.
                                if label not in current_labels:
                                    current_labels.append(label)
                                # Sort the labels to enforce a canonical order.
                                labels[i] = "+".join(sorted(current_labels))
                processed.append({"tokens": tokens, "labels": labels})
        return processed

    def get_annotations(self):
        """
        Returns the list of processed annotations.
        """
        return self.annotations

# -------------------------------
# Fuzzy Matching Utilities
# -------------------------------

def extract_spans(tokens, labels, target_label):
    """
    Extract tokens that contain the target label.
    """
    spans = []
    current_span = []

    for token, label in zip(tokens, labels):
        if target_label in label:  # Check if target label exists
            current_span.append(token)
        else:
            if current_span:
                spans.append(current_span)
                current_span = []

    if current_span:
        spans.append(current_span)

    return spans

def fuzzy_match_score(span1, span2):
    """
    Computes the fuzzy matching score between two spans.
    Converts the nested span lists into flat lists before computing set similarity.
    """
    if not span1 and not span2:
        return 1.0  # Both empty = perfect match
    if not span1 or not span2:
        return 0.0  # One empty, one not = full disagreement

    # Flatten spans before applying set operations
    set1, set2 = set(sum(span1, [])), set(sum(span2, []))
    overlap = set1 & set2

    return len(overlap) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0.0

def fuzzy_distance(span1, span2):
    """
    Converts the fuzzy match score into a distance metric.
    A perfect match gives 0 distance; no overlap gives 1.
    """
    return 1 - fuzzy_match_score(span1, span2)

def average_pairwise_distance(spans):
    """
    Computes the average pairwise fuzzy distance for a list of spans.
    """
    distances = [
        fuzzy_distance(spans[i], spans[j])
        for i, j in itertools.combinations(range(len(spans)), 2)
    ]
    return sum(distances) / len(distances) if distances else 0.0

# -------------------------------
# Krippendorff's Alpha with Fuzzy Matching
# -------------------------------

class FuzzyKrippendorff:
    def __init__(self, annotation_processors, targets=("cause", "effect")):
        """
        Initializes the calculator with a list of AnnotationProcessor instances.
        'targets' indicates which labels (e.g., "cause" and "effect") to consider.
        """
        self.annotation_processors = annotation_processors
        self.targets = targets
        self.annotations_list = [proc.get_annotations() for proc in self.annotation_processors]
        self.num_sentences = len(self.annotations_list[0])

    def compute_observed_disagreement(self, target_label):
        """
        Computes the observed disagreement (D_o) across all annotators for a specific label.
        """
        observed_distances = []

        for idx in range(self.num_sentences):
            tokens_list = [annotations[idx]["tokens"] for annotations in self.annotations_list]
            labels_list = [annotations[idx]["labels"] for annotations in self.annotations_list]

            spans = [extract_spans(tokens, labels, target_label)
                     for tokens, labels in zip(tokens_list, labels_list)]

            distances = [
                fuzzy_distance(spans[i], spans[j])
                for i, j in itertools.combinations(range(len(spans)), 2)
            ]

            if distances:
                observed_distances.append(sum(distances) / len(distances))

        return sum(observed_distances) / len(observed_distances) if observed_distances else 0.0

    def compute_expected_disagreement(self, target_label):
        """
        Computes the expected disagreement (D_e) by pooling all spans for a target label
        across all sentences and annotators.
        """
        all_spans = []

        for annotations in self.annotations_list:
            for sentence in annotations:
                span = extract_spans(sentence["tokens"], sentence["labels"], target_label)
                all_spans.append(span)

        return average_pairwise_distance(all_spans)

    def compute_krippendorff_alpha(self):
        """
        Computes Krippendorff's alpha using fuzzy matching for multiple labels.
        """
        alphas = []
        for target in self.targets:
            D_o = self.compute_observed_disagreement(target)
            D_e = self.compute_expected_disagreement(target)
            alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
            alphas.append(alpha)

        return sum(alphas) / len(alphas)



# -------------------------------
# Example Usage
# -------------------------------
"""
if __name__ == "__main__":
    # Define file paths for the three annotators.
    paths = {
        "rasoul": 'rasoul.jsonl',
        "caspar": 'caspar.jsonl',
        "bennett": 'bennett.jsonl'
    }


    # Initialize an AnnotationProcessor for each file.
    processors = [AnnotationProcessor(path) for path in paths.values()]

    # Create a FuzzyKrippendorff instance (considering both "cause" and "effect" targets).
    fk = FuzzyKrippendorff(annotation_processors=processors, targets=("cause", "effect"))

    # Compute and display Krippendorff's alpha.
    alpha = fk.compute_krippendorff_alpha()
    print("Krippendorff's alpha (with fuzzy matching):", alpha)
"""