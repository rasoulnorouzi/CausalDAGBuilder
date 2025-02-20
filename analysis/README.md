# Fuzzy Krippendorff's Alpha

Let α be the Fuzzy Krippendorff's Alpha coefficient defined as:

α = 1 - (D_o / D_e)                                                               (1)

where D_o is observed disagreement and D_e is expected disagreement.

For annotation spans S₁ and S₂, the fuzzy match score F is:

F(S₁, S₂) = |S₁ ∩ S₂| / min(|S₁|, |S₂|)                                         (2)

The fuzzy distance d between spans:

d(S₁, S₂) = 1 - F(S₁, S₂)                                                       (3)

For N sentences and M annotators, observed disagreement D_o:

D_o = (1/N) ∑ᵢ₌₁ᴺ δᵢ                                                            (4)

where sentence-level disagreement δᵢ:

δᵢ = (2/M(M-1)) ∑ₖ₌₁ᴹ⁻¹ ∑ₗ₌ₖ₊₁ᴹ d(Sᵢₖ, Sᵢₗ)                                    (5)

Expected disagreement D_e for pooled spans Π = {Sᵢₖ | i ∈ [1,N], k ∈ [1,M]}:

D_e = (2/|Π|(|Π|-1)) ∑ₐ₌₁|Π|⁻¹ ∑ᵦ₌ₐ₊₁|Π| d(Sₐ, Sᵦ)                            (6)

For L labels, final alpha:

α = (1/L) ∑ᵢ₌₁ᴸ max(0, 1 - D_oᵢ/D_eᵢ)                                          (7)

**Algorithm 1**: Fuzzy Krippendorff's Alpha Computation
```
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
