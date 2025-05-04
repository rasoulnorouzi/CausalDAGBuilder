#%%
import causal_eval
import os

gold_path = r"datasets\training_test_samples_gold\doccano_test.jsonl"
pred_path = r"weak_supervision/dataset/llama3_8b_output_test.jsonl"

# Evaluate scenario A
performance_A = causal_eval.evaluate(
    gold_path=gold_path,
    pred_path=pred_path,
    scenario_task2="A"
)
causal_eval.display_results(performance_A, "A")

# Evaluate scenario B
performance_B = causal_eval.evaluate(
    gold_path=gold_path,
    pred_path=pred_path,
    scenario_task2="B"
)
causal_eval.display_results(performance_B, "B")

