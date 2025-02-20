### 1. **Fuzzy Matching for Annotation Spans**
When comparing annotations, we often need to decide whether two spans of text "match" even if they are not exactly identical. The approach here is to use **fuzzy matching**. Given two spans of tokens (which may be lists of tokens), we first **flatten** the spans into sets of tokens and then measure their overlap.

#### **Fuzzy Match Score**
Let:
- S₁ and S₂ be the flattened sets of tokens extracted from the annotation spans.
- |S₁| and |S₂| be the number of unique tokens in each span.
- S₁ ∩ S₂ be the set of tokens common to both.

Then the **fuzzy match score** is defined as:

```
FuzzyMatchScore(S₁, S₂) = |S₁ ∩ S₂| / min{|S₁|, |S₂|}
```

This score lies in the interval [0,1], where 1 indicates a perfect match and 0 indicates no overlap.

#### **Fuzzy Distance**
We then define a **distance metric** based on the fuzzy match score:

```
d(S₁, S₂) = 1 - FuzzyMatchScore(S₁, S₂)
```

A perfect match gives a distance of 0, and no overlap gives a distance of 1.

### 2. **Observed and Expected Disagreement**

#### **Observed Disagreement (Dₒ)**
For a given target label (e.g., "cause" or "effect"), suppose we have n annotators. For each sentence, we extract the corresponding spans from each annotator. For every unique pair of annotators (i, j), we compute the fuzzy distance between their spans:

```
dᵢⱼ(s) = d(Sᵢ(s), Sⱼ(s))
```

where Sᵢ(s) is the span extracted by annotator i in sentence s.

For each sentence s, the average pairwise distance is:

```
Dₒ(s) = (2 / (n(n-1))) ∑ᵢ<ⱼ dᵢⱼ(s)
```

Then, averaging over all N sentences:

```
Dₒ = (1/N) ∑ₛ₌₁ᴺ Dₒ(s)
```

#### **Expected Disagreement (Dₑ)**
Instead of comparing annotations sentence by sentence, Dₑ is computed by pooling all spans for the target label across sentences and annotators. Denote the pooled set of spans by {S₁, S₂, ..., Sₘ} (where M is the total number of spans).

The average pairwise fuzzy distance over all pairs is:

```
Dₑ = (2 / (M(M-1))) ∑ᵢ<ⱼ d(Sᵢ, Sⱼ)
```

### 3. **Krippendorff's Alpha**
Krippendorff's alpha is then defined by comparing the observed disagreement Dₒ with the expected disagreement Dₑ:

```
α = 1 - (Dₒ/Dₑ)    if Dₑ > 0
```

If Dₑ = 0 (i.e., if there is no expected disagreement because annotations are completely uniform), we set α = 1.0 by definition. Note that if the computed α falls below 0, it is often truncated to 0.
