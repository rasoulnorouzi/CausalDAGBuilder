### 1. **Fuzzy Matching for Annotation Spans**

When comparing annotations, we often need to decide whether two spans of text “match” even if they are not exactly identical. The approach here is to use **fuzzy matching**. Given two spans of tokens (which may be lists of tokens), we first **flatten** the spans into sets of tokens and then measure their overlap.

#### **Fuzzy Match Score**

Let:
- $S_1$ and $S_2$ be the flattened sets of tokens extracted from the annotation spans.
- $|S_1|$ and $|S_2|$ be the number of unique tokens in each span.
- $S_1 \cap S_2$ be the set of tokens common to both.

Then the **fuzzy match score** is defined as:

$$
\text{FuzzyMatchScore}(S_1, S_2) = \frac{|S_1 \cap S_2|}{\min\{|S_1|, |S_2|\}}
$$

This score lies in the interval $[0,1]$, where 1 indicates a perfect match and 0 indicates no overlap.

#### **Fuzzy Distance**

We then define a **distance metric** based on the fuzzy match score:

$$
d(S_1, S_2) = 1 - \text{FuzzyMatchScore}(S_1, S_2)
$$

A perfect match gives a distance of 0, and no overlap gives a distance of 1.

### 2. **Observed and Expected Disagreement**

#### **Observed Disagreement ($D_o$)**

For a given target label (e.g., "cause" or "effect"), suppose we have $n$ annotators. For each sentence, we extract the corresponding spans from each annotator. For every unique pair of annotators $(i, j)$, we compute the fuzzy distance between their spans:

$$
d_{ij}(s) = d\big(S_i(s), S_j(s)\big)
$$

where $S_i(s)$ is the span extracted by annotator $i$ in sentence $s$.

For each sentence $s$, the average pairwise distance is:

$$
D_o(s) = \frac{2}{n(n-1)} \sum_{i<j} d_{ij}(s)
$$

Then, averaging over all $N$ sentences:

$$
D_o = \frac{1}{N} \sum_{s=1}^{N} D_o(s)
$$

#### **Expected Disagreement ($D_e$)**

Instead of comparing annotations sentence by sentence, $D_e$ is computed by pooling all spans for the target label across sentences and annotators. Denote the pooled set of spans by $\{S_1, S_2, \ldots, S_M\}$ (where $M$ is the total number of spans).

The average pairwise fuzzy distance over all pairs is:

$$
D_e = \frac{2}{M(M-1)} \sum_{i<j} d(S_i, S_j)
$$

### 3. **Krippendorff’s Alpha**

Krippendorff’s alpha is then defined by comparing the observed disagreement $D_o$ with the expected disagreement $D_e$:

$$
\alpha = 1 - \frac{D_o}{D_e} \quad \text{if } D_e > 0
$$

If $D_e = 0$ (i.e., if there is no expected disagreement because annotations are completely uniform), we set $\alpha = 1.0$ by definition. Note that if the computed $\alpha$ falls below 0, it is often truncated to 0.

