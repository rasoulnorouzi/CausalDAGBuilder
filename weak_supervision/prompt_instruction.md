# Objective

Analyze the input sentence to determine if it describes a causal relationship. If it is causal, identify all cause-effect pairs and their polarity. If not causal, state so.

# IMPORTANT: Required Output Format

You MUST strictly adhere to this exact output format:

For causal sentences:

```
"sentence": [exact input sentence]
"causal": yes,
"causes": {
    [cause_1]: {
        [effect_1]: [polarity],
        [effect_2]: [polarity]
    },
    [cause_2]: {
        [effect_1]: [polarity]
    }
}

```

For non-causal sentences:

```
"sentence": [exact input sentence]
"causal": no,
"causes": {}

```

DO NOT deviate from this format. Use the exact format structure shown above.

# Definitions

- **Causal Sentence**: A sentence stating that one event/condition (Cause) directly leads to, produces, affects, requires, or results in another event/condition (Effect). Look for words like "leads to," "causes," "results in," "affects," "requires," "because," "due to," "therefore," "producing."
- **Non-Causal Sentence**: Describes association, correlation, co-occurrence, comparison, or sequences without a direct causal link or is a question sentence. Output "causal": no.
- **Cause**: The event or condition that initiates the effect. Extract the exact phrase from the sentence.
- **Effect**: The outcome resulting from the cause. Extract the exact phrase from the sentence.
- **Polarity**: The nature of the causal link:
    - **Positive**: The cause increases or intensifies the effect (e.g., "leads to more aggressive responses").
    - **Negative**: The cause decreases or diminishes the effect (e.g., "reduces errors").
    - **Neutral**: A causal link exists, but direction (increase/decrease) isn't specified or clearly implied (e.g., "affects the ability").
    - **Zero**: The sentence explicitly states a lack of causal connection between a specific cause and effect, often using negation (e.g., "did not moderate reactions," "but not the ability to initiate").

# Causal Requirement Rules

- A requirement is causal if its absence prevents an effect (necessary) and, when present alone, guarantees the effect (sufficient).
- Conversely, if an effect can occur without the requirement, it is non-causal.
- In a statement like "A causes B by/through/because of C," treat C as the initial necessary factor that produces B, which in turn is required for A to occur; each link in the chain should be evaluated for its necessary and sufficient role.
- If replacing "when" with "if" or "because" preserves the meaning of one event triggering another, then "when" signals a causal relationship; if not, it merely marks a temporal sequence.

# Instructions

1. Read the input sentence carefully. Ignore parenthetical citations like (Author, Year) or page numbers (p.XXX).
2. Determine if the sentence describes at least one causal relationship. Set "causal" to "yes" or "no".
3. If "causal": yes:
    - Identify all distinct cause-effect pairs presented in the sentence.
    - A single cause might have multiple effects. List each effect separately under the same cause.
    - A single effect might have multiple causes. List each cause separately leading to the same effect.
    - A sentence can contain multiple independent cause-effect pairs. List them all.
    - An effect can sometimes become the cause of a subsequent effect (causal chain). List both pairs sequentially.
    - For each pair, extract the exact Cause phrase and the exact Effect phrase from the sentence. Do not modify, shorten, or extend the phrases. This is true for cause spans as well as effect spans.
    - Determine the Polarity (Positive, Negative, Neutral, Zero) for each pair based on the definitions above.
    - Format the output exactly as shown in the required output format.
4. If "causal": no:
    - Output the format for non-causal sentences as shown above.

# Examples

```
Input: according to smith (2008), there is a need for substantial and consistent support and encouragement for women to participate in studies.
Output:
"sentence": "according to smith (2008), there is a need for substantial and consistent support and encouragement for women to participate in studies."
"causal": yes,
"causes": {
"substantial and consistent support": {
"women to participate in studies": "Neutral"
},
"encouragement": {
"women to participate in studies": "Neutral"
}
}
```

```
Input: The ostracism alarm comes in the form of feeling social pain, resulting in negative affect and threats to basic needs.
Output:
"sentence": "The ostracism alarm comes in the form of feeling social pain, resulting in negative affect and threats to basic needs."
"causal": yes,
"causes": {
"feeling social pain": {
"negative affect": "Neutral",
"threats to basic needs": "Neutral"
}
}
```

```
Input: unexpected rejection leads to more aggressive responses.
Output:
"sentence": "unexpected rejection leads to more aggressive responses."
"causal": yes,
"causes": {
"unexpected rejection": {
"more aggressive responses": "Positive"
}
}
```

```
Input: based on my findings with the tracing task, it appears that ostracism affects the ability to persist at a difficult task, but not the ability to initiate the task.
Output:
"sentence": "based on my findings with the tracing task, it appears that ostracism affects the ability to persist at a difficult task, but not the ability to initiate the task."
"causal": yes,
"causes": {
"ostracism": {
"ability to persist at a difficult task": "Neutral",
"ability to initiate the task": "Zero"
}
}
```

```
Input: consistent with our finding, leary, haupt, strausser, and chokel (1998) reported that trait self-esteem did not moderate participants' reactions to interpersonal rejection.
Output:
"sentence": "consistent with our finding, leary, haupt, strausser, and chokel (1998) reported that trait self-esteem did not moderate participants' reactions to interpersonal rejection."
"causal": yes,
"causes": {
"trait self-esteem": {
"participants' reactions to interpersonal rejection": "Zero"
}
}
```

```
Input: the present research tests the hypothesis that, because of this unwillingness to regulate the self, excluded (relative to included or control) participants are more likely to exhibit the confirmation bias (fischer, Greitemeyer, & frey, 2008).
Output:
"sentence": "the present research tests the hypothesis that, because of this unwillingness to regulate the self, excluded (relative to included or control) participants are more likely to exhibit the confirmation bias (fischer, Greitemeyer, & frey, 2008)."
"causal": yes,
"causes": {
"this unwillingness to regulate the self": {
"excluded (relative to included or control) participants are more likely to exhibit the confirmation bias": "Positive"
}
}
```

```
Input: Furthermore, it is believed that the pursuit of selfesteem lies in an individuals need to manage their anxieties and fears (Crocker & Park, 2004; Greenberg 6 et al., 1992).
Output:
"sentence": "Furthermore, it is believed that the pursuit of selfesteem lies in an individuals need to manage their anxieties and fears (Crocker & Park, 2004; Greenberg 6 et al., 1992)."
"causal": no,
"causes": {}
```

```
Input: the drops in performance we did observe among ostracized participants could be stronger with less interesting tasks; a more interesting task might serve as a more appealing means to recover from the ostracism experience than a boring task would.
Output:
"sentence": "the drops in performance we did observe among ostracized participants could be stronger with less interesting tasks; a more interesting task might serve as a more appealing means to recover from the ostracism experience than a boring task would."
"causal": yes,
"causes": {
"less interesting tasks": {
"stronger drops in performance we did observe among ostracized participants": "Positive"
},
"a more interesting task": {
"serve as a more appealing means to recover from the ostracism experience": "Positive"
}
}
```

```
Input: because each instance of qualitative research is perceived as a unique process requiring the researcher to craft his or her own method, flexibility, versatility, and creativity have been emphasised, methodological ambiguity tolerated as an inescapable, even desirable, component of the process.
Output:
"sentence": "because each instance of qualitative research is perceived as a unique process requiring the researcher to craft his or her own method, flexibility, versatility, and creativity have been emphasised, methodological ambiguity tolerated as an inescapable, even desirable, component of the process."
"causal": yes,
"causes": {
"each instance of qualitative research is perceived as a unique process requiring the researcher to craft his or her own method": {
"flexibility, versatility, and creativity have been emphasised": "Neutral",
"methodological ambiguity tolerated as an inescapable, even desirable, component of the process": "Neutral"
}
}
```

```
Input: these data fully support our main hypothesis, and show not only that self affirmation can facilitate non-defensive processing among unrealistic optimists but that its absence in the face of threat can foster such defensive processing.
Output:
"sentence": "these data fully support our main hypothesis, and show not only that self affirmation can facilitate non-defensive processing among unrealistic optimists but that its absence in the face of threat can foster such defensive processing."
"causal": yes,
"causes": {
"self affirmation": {
"facilitate non-defensive processing among unrealistic optimists": "Positive"
},
"its absence in the face of threat": {
"foster such defensive processing": "Positive"
}
}
```

```
Input: the cluster analysis results suggest that the built environment is the outcome of mode of governance producing places and contradictions.
Output:
"sentence": "the cluster analysis results suggest that the built environment is the outcome of mode of governance producing places and contradictions."
"causal": yes,
"causes": {
"mode of governance": {
"the built environment": "Neutral"
},
"the built environment": {
"producing places and contradictions": "Neutral"
}
}
```

```
Input: ethnographers might become an advocate when they become ""aware of an issue"" through their research or when they become ""more deeply committed to the issue"" (p.151) through their research.
Output:
"sentence": "ethnographers might become an advocate when they become ""aware of an issue"" through their research or when they become ""more deeply committed to the issue"" (p.151) through their research."
"causal": yes,
"causes": {
"become ""aware of an issue"" through their research": {
"ethnographers might become an advocate": "Neutral"
},
"become ""more deeply committed to the issue"" through their research": {
"ethnographers might become an advocate": "Neutral"
}
}
```

```
Input: Accumulated total earnings are negatively correlated with the likelihood of repayment.
Output:
"sentence": "Accumulated total earnings are negatively correlated with the likelihood of repayment."
"causal": no,
"causes": {}
```

```
Input: producing valid and relevant information therefore requires organisation of the in‐flow of information as well as a degree of critical distance kept with the field intensity.
Output:
"sentence": "producing valid and relevant information therefore requires organisation of the in‐flow of information as well as a degree of critical distance kept with the field intensity."
"causal": yes,
"causes": {
"organisation of the in‐flow of information": {
"producing valid and relevant information": "Neutral"
},
"a degree of critical distance kept with the field intensity": {
"producing valid and relevant information": "Neutral"
}
}
```

IMPORTANT: You MUST use the exact format specified above
You MUST generate output in the required format shown in the examples. Do not add explanations, reasoning, or any content outside of the required format.

Now, analyze the following sentence:
[Input Sentence Here]