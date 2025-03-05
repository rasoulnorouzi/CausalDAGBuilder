import json
import spacy
import itertools

class KrippendorffSpanMatcher:
    def __init__(self, annotator_paths, language="en", matching_mode="fuzzy", targets=("cause", "effect", "cause+effect")):
        """
        annotator_paths: dict mapping annotator names to file paths.
        language: spaCy language model to use.
        matching_mode: "fuzzy" (proportional token overlap) or "overlap" (any overlap = full agreement).
        targets: tuple of target labels to consider.
        """
        self.annotator_paths = annotator_paths
        self.annotator_names = list(annotator_paths.keys())
        self.language = language
        self.matching_mode = matching_mode
        self.targets = targets

        spacy.prefer_gpu()
        self.nlp = spacy.blank(language)
        # Process all annotator files once and store annotations.
        self.annotations_list = [self._process_annotations(path) for path in annotator_paths.values()]
        self.num_sentences = len(self.annotations_list[0])
    
    def _process_annotations(self, file_path):
        """
        Processes a JSONL annotation file.
        Each record is converted into a dictionary with:
          - 'tokens': tokenized words of the sentence.
          - 'labels': labels for each token (or "NONE" if no relevant entity is found).
        """
        processed = []
        with open(file_path, "r") as file:
            for line in file:
                record = json.loads(line)
                sentence = record.get("text", "")
                entities = record.get("entities", [])
                doc = self.nlp(sentence)
                tokens = [token.text for token in doc]
                labels = ["NONE"] * len(tokens)
                
                # If no entities or all are "non-causal", label all tokens as "NONE".
                if not entities or all(entity.get("label") == "non-causal" for entity in entities):
                    processed.append({"tokens": tokens, "labels": labels})
                    continue
                
                for entity in entities:
                    label = entity.get("label")
                    start_offset = entity.get("start_offset")
                    end_offset = entity.get("end_offset")
                    for i, token in enumerate(doc):
                        # If token overlaps with the entity span:
                        if not (token.idx + len(token.text) <= start_offset or token.idx >= end_offset):
                            if labels[i] == "NONE":
                                labels[i] = label
                            else:
                                # Split and add new label if not already present.
                                current_labels = labels[i].split("+")
                                if label not in current_labels:
                                    current_labels.append(label)
                                labels[i] = "+".join(sorted(current_labels))
                processed.append({"tokens": tokens, "labels": labels})
        return processed

    def extract_spans(self, tokens, labels, target_label):
        """
        Extracts continuous spans (lists of tokens) for which the token label includes target_label.
        """
        spans = []
        current_span = []
        for token, label in zip(tokens, labels):
            if target_label in label:
                current_span.append(token)
            else:
                if current_span:
                    spans.append(current_span)
                    current_span = []
        if current_span:
            spans.append(current_span)
        return spans

    def match_score(self, span1, span2):
        """
        Computes the matching score between two spans.
        - In "fuzzy" mode, the score is proportional to the token overlap:
              len(overlap) / min(len(set(span1)), len(set(span2))).
        - In "overlap" mode, any overlap returns 1.0 (full agreement); no overlap returns 0.0.
        Both span1 and span2 are expected to be lists of spans.
        """
        if not span1 and not span2:
            return 1.0  # Both empty = perfect match
        if not span1 or not span2:
            return 0.0  # One empty, one not = disagreement

        # Flatten the lists of spans into sets of tokens.
        set1 = set(sum(span1, []))
        set2 = set(sum(span2, []))
        
        if self.matching_mode == "overlap":
            return 1.0 if set1 & set2 else 0.0
        else:  # "fuzzy" mode
            overlap = set1 & set2
            return len(overlap) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0.0

    def fuzzy_distance(self, span1, span2):
        """
        Converts the match score into a distance metric.
        Perfect match gives a distance of 0; no overlap gives 1.
        """
        return 1 - self.match_score(span1, span2)

    def average_pairwise_distance(self, spans):
        """
        Computes the average pairwise fuzzy distance for a list of spans.
        """
        distances = [
            self.fuzzy_distance(spans[i], spans[j])
            for i, j in itertools.combinations(range(len(spans)), 2)
        ]
        return sum(distances) / len(distances) if distances else 0.0

    def compute_observed_disagreement(self, target_label):
        """
        Computes the observed disagreement (D_o) across all annotators for a given target label.
        """
        observed_distances = []
        for idx in range(self.num_sentences):
            tokens_list = [annotations[idx]["tokens"] for annotations in self.annotations_list]
            labels_list = [annotations[idx]["labels"] for annotations in self.annotations_list]
            spans = [self.extract_spans(tokens, labels, target_label)
                     for tokens, labels in zip(tokens_list, labels_list)]
            distances = [
                self.fuzzy_distance(spans[i], spans[j])
                for i, j in itertools.combinations(range(len(spans)), 2)
            ]
            if distances:
                observed_distances.append(sum(distances) / len(distances))
        return sum(observed_distances) / len(observed_distances) if observed_distances else 0.0

    def compute_expected_disagreement(self, target_label):
        """
        Computes the expected disagreement (D_e) by pooling all spans across all sentences and annotators.
        """
        all_spans = []
        for annotations in self.annotations_list:
            for sentence in annotations:
                span = self.extract_spans(sentence["tokens"], sentence["labels"], target_label)
                all_spans.append(span)
        return self.average_pairwise_distance(all_spans)

    def compute_krippendorff_alpha(self):
        """
        Computes Krippendorff's alpha (averaged over the specified target labels) using the fuzzy matching approach.
        """
        alphas = []
        for target in self.targets:
            D_o = self.compute_observed_disagreement(target)
            D_e = self.compute_expected_disagreement(target)
            alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
            alphas.append(max(alpha, 0.0))
        return sum(alphas) / len(alphas)
    
    def get_unique_labels(self):
        """
        Computes the unique set of labels for each annotator.
        Returns a dictionary mapping annotator names to their set of unique labels.
        """
        unique_labels = {}
        for name, annotations in zip(self.annotator_names, self.annotations_list):
            labels_set = set()
            for sentence in annotations:
                labels_set.update(sentence["labels"])
            unique_labels[name] = labels_set
        return unique_labels


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Define file paths for the annotators.
    paths = {
        "rasoul": 'rasoul.jsonl',
        "caspar": 'caspar.jsonl',
        "bennett": 'bennett.jsonl'
    }
    
    
    
    # Report the unique labels for each annotator.
    unique_labels = matcher.get_unique_labels()
    for name, labels in unique_labels.items():
        print(f"{name}: {labels}")
        # Expected outputs:
        # rasoul: {'NONE', 'effect', 'cause', 'cause+effect'}
        # caspar: {'NONE', 'effect', 'cause', 'cause+effect'}
        # bennett: {'NONE', 'effect', 'cause'}

    print("---")
    # matching_mode set to "overlap"
    overlap_matcher = KrippendorffSpanMatcher(annotator_paths=paths, matching_mode="overlap", targets=("cause", "effect"))
    # Compute and display overlap Krippendorff's alpha.
    overlap_alpha = overlap_matcher.compute_krippendorff_alpha()
    print("Overlap Krippendorff's alpha:", overlap_alpha)

    print("---")	
    # matching_mode set to "fuzzy"
    fuzzy_matcher = KrippendorffSpanMatcher(annotator_paths=paths, matching_mode="fuzzy", targets=("cause", "effect"))
    # Compute and display fuzzy Krippendorff's alpha.
    fuzzy_alpha = fuzzy_matcher.compute_krippendorff_alpha()
    print("Fuzzy Krippendorff's alpha:", fuzzy_alpha)

