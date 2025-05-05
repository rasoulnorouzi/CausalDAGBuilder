"""joint_causal_ie.py
End‑to‑end training script for the gold‑only setting.
• Uses a single RoBERTa‑base encoder and three lightweight heads.
• Computes automatic class weights from the training split.
• Handles negative sampling (R = 4) for relation training.
• Exports predictions back to Doccano JSONL for evaluation via causal_eval.py.

NOTE: weak‑supervision paths are commented but left in place for future use.
"""

import argparse
import pathlib
from data import load_jsonl, CausalDataset, collate
from model import JointCausalIE
from train import run_train
from predict import predict_to_jsonl

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("train_jsonl"); parser.add_argument("val_jsonl"); parser.add_argument("test_jsonl")
    args=parser.parse_args()
    train_recs=load_jsonl(pathlib.Path(args.train_jsonl))
    val_recs  =load_jsonl(pathlib.Path(args.val_jsonl))
    test_recs =load_jsonl(pathlib.Path(args.test_jsonl))
    # Use only 10 samples for training
    train_recs = train_recs[:10]
    ds_train=CausalDataset(train_recs,"train"); ds_val=CausalDataset(val_recs,"dev"); ds_test=CausalDataset(test_recs,"test")
    model = JointCausalIE()
    run_train(model, ds_train, ds_val, collate)
    predict_to_jsonl(model, ds_test, "gold_joint.ckpt", pathlib.Path("pred_test.jsonl"))
    # Show a few lines of the prediction output
    print("\nSample predictions from pred_test.jsonl:")
    with open("pred_test.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i >= 4:
                break
    # Evaluate predictions using causal_eval.py
    import causal_eval
    results = causal_eval.evaluate("datasets/doccano_test.jsonl", "pred_test.jsonl")
    causal_eval.display_results(results, scenario="A")
    causal_eval.display_results(results, scenario="B")
