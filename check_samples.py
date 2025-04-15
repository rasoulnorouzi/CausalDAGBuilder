def get_present_samples(prompt_samples, texts):
    """
    Returns a list of samples from prompt_samples that are present in texts.
    """
    texts_set = set(texts)
    return [sample for sample in prompt_samples if sample in texts_set]

# Example usage:
# prompt_samples = ["sample1", "sample2", "sample3"]
# texts = ["sample2", "sample4", "sample1"]
# present = get_present_samples(prompt_samples, texts)
# print(present)  # Output: ['sample1', 'sample2']
