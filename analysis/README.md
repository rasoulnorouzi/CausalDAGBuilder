# Fuzzy Krippendorff's Alpha

## Mathematical Formulation

### Core Formula
Let α be the Fuzzy Krippendorff's Alpha coefficient:

```math
α = 1 - \frac{D_o}{D_e}  \tag{1}
```

where D_o is observed disagreement and D_e is expected disagreement.

### Fuzzy Match Score
For annotation spans S₁ and S₂:

```math
F(S_1, S_2) = \frac{|S_1 \cap S_2|}{\min(|S_1|, |S_2|)}  \tag{2}
```

### Fuzzy Distance
Distance between spans:

```math
d(S_1, S_2) = 1 - F(S_1, S_2)  \tag{3}
```

### Observed Disagreement
For N sentences and M annotators:

```math
D_o = \frac{1}{N} \sum_{i=1}^N δ_i  \tag{4}
```

where sentence-level disagreement δᵢ:

```math
δ_i = \frac{2}{M(M-1)} \sum_{k=1}^{M-1} \sum_{l=k+1}^M d(S_{ik}, S_{il})  \tag{5}
```

### Expected Disagreement
For pooled spans Π = {Sᵢₖ | i ∈ [1,N], k ∈ [1,M]}:

```math
D_e = \frac{2}{|Π|(|Π|-1)} \sum_{a=1}^{|Π|-1} \sum_{β=a+1}^{|Π|} d(S_a, S_β)  \tag{6}
```

### Multi-Label Extension
For L labels, final alpha:

```math
α = \frac{1}{L} \sum_{i=1}^L \max(0, 1 - \frac{D_{oi}}{D_{ei}})  \tag{7}
```

## Algorithm

```
Algorithm 1: Fuzzy Krippendorff's Alpha Computation

Input: A = {a₁, ..., aₘ} annotators' spans for N sentences
Output: α (Fuzzy Krippendorff's Alpha coefficient)

Function ComputeFuzzyKrippendorff(A):
    D_o ← 0
    for i ← 1 to N do
        δ ← 0
        for k ← 1 to M-1 do
            for l ← k+1 to M do
                δ ← δ + Distance(A[k][i], A[l][i])
            end for
        end for
        D_o ← D_o + (2δ)/(M(M-1))
    end for
    D_o ← D_o/N
    
    Π ← Pool(A)
    D_e ← 0
    for a ← 1 to |Π|-1 do
        for b ← a+1 to |Π| do
            D_e ← D_e + Distance(Π[a], Π[b])
        end for
    end for
    D_e ← (2D_e)/(|Π|(|Π|-1))
    
    return 1 - D_o/D_e

Function Distance(S₁, S₂):
    if S₁ = ∅ and S₂ = ∅ then return 0
    if S₁ = ∅ or S₂ = ∅ then return 1
    return 1 - |S₁ ∩ S₂|/min(|S₁|, |S₂|)
```

### Notes:
- All formulas use standard set theory notation
- Empty set comparisons are handled as special cases
- Algorithm complexity: O(N·M² + |Π|²)
