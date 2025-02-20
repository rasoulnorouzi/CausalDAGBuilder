# Mathematical Formulation of Fuzzy Krippendorff's Alpha

## 1. Core Formula

The Fuzzy Krippendorff's Alpha (α) is defined as:

α = 1 - (D_o / D_e)

Where:
- D_o is the observed disagreement
- D_e is the expected disagreement

## 2. Fuzzy Set Operations

### 2.1 Fuzzy Match Score

For two annotation spans S₁ and S₂, the fuzzy match score is defined as:

F(S₁, S₂) = |S₁ ∩ S₂| / min(|S₁|, |S₂|)

Special cases:
- F(∅, ∅) = 1 (empty sets are considered perfect matches)
- F(S₁, ∅) = F(∅, S₂) = 0 (empty set and non-empty set have no match)

### 2.2 Fuzzy Distance Metric

The fuzzy distance d between two spans is defined as:

d(S₁, S₂) = 1 - F(S₁, S₂)

Properties:
- d(S₁, S₂) ∈ [0,1]
- d(S₁, S₁) = 0 (identity)
- d(S₁, S₂) = d(S₂, S₁) (symmetry)

## 3. Observed Disagreement (D_o)

For a corpus with N sentences and M annotators:

D_o = (1/N) ∑ᵢ₌₁ᴺ δᵢ

Where δᵢ for each sentence i is:

δᵢ = (2/M(M-1)) ∑ₖ₌₁ᴹ⁻¹ ∑ₗ₌ₖ₊₁ᴹ d(Sᵢₖ, Sᵢₗ)

Where:
- Sᵢₖ is the span annotation from annotator k for sentence i
- d(Sᵢₖ, Sᵢₗ) is the fuzzy distance between two annotations

## 4. Expected Disagreement (D_e)

Let Π be the set of all spans across all sentences and annotators:

Π = {Sᵢₖ | i ∈ [1,N], k ∈ [1,M]}

Then:

D_e = (2/|Π|(|Π|-1)) ∑ₐ₌₁|Π|⁻¹ ∑ᵦ₌ₐ₊₁|Π| d(Sₐ, Sᵦ)

Where:
- |Π| is the total number of spans in the pooled set
- Sₐ, Sᵦ are any two spans from the pooled set

## 5. Multi-Label Extension

For a set of L target labels (e.g., "cause", "effect"), the final alpha is:

α = (1/L) ∑ᵢ₌₁ᴸ αᵢ

Where αᵢ is the alpha value computed for each label i:

αᵢ = max(0, 1 - D_oᵢ/D_eᵢ)

The max function ensures non-negative alpha values.

## 6. Properties

1. Range: α ∈ [0,1]
2. Perfect Agreement: α = 1 when D_o = 0
3. Chance Agreement: α = 0 when D_o = D_e
4. For each label i: αᵢ ≥ 0

## 7. Interpretation Guidelines

- α > 0.8: Strong agreement
- 0.67 < α ≤ 0.8: Substantial agreement
- 0.4 < α ≤ 0.67: Moderate agreement
- α ≤ 0.4: Poor agreement

These thresholds may vary based on the specific annotation task and requirements.
