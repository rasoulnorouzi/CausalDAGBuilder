import math
import random
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from collections import Counter
import torch.nn as nn
import itertools  # Added for loss_fn

LAMBDAS = dict(cls=1.0, span=2.0, rel=3.0)
LR = 2e-5
EPOCHS = 1
BATCH_SIZE = 16
GRAD_ACCUM = 2
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0
NEG_RATIO = 4
PAIR_WINDOW = 50
SEED = 8642
random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class weights

def compute_class_weights(records):
    token_counts = Counter()
    pol_counts = Counter()
    for r in records:
        token_counts[0] += len(r["text"])
        for _,_,l in r["entities"]:
            token_counts[l] += 1
        for _,_,pol in r["relations"]:
            pol_counts[pol] += 1
    w_O = 1 - (token_counts[0] / sum(token_counts.values()))
    w_span = torch.tensor([w_O,1,1,1,1], dtype=torch.float, device=device)
    rel_labels = ["positive","negative","zero","neutral","none"]
    w_rel = torch.tensor([1/math.log1p(pol_counts[x] or 1) for x in rel_labels], device=device)
    return w_span, w_rel

# Loss function
ce = nn.CrossEntropyLoss(reduction="none")

def loss_fn(batch, outputs, w_span, w_rel):
    logits_cls, logits_span, logits_rel = outputs
    l1 = ce(logits_cls, batch["cls_labels"].to(device)).mean()
    span_gold = batch["bio_labels"].to(device)
    ls = ce(logits_span.permute(0,2,1), span_gold)
    ls = (ls * w_span[span_gold]).mean()
    br = batch["batch_meta"]
    rel_losses = []
    for b_idx, meta in enumerate(br):
        cause_idx = meta["cause_idx"]
        effect_idx = meta["effect_idx"]
        pos = set()
        for ci in cause_idx:
            for ei in effect_idx:
                if abs(ci-ei) <= PAIR_WINDOW:
                    pos.add((ci,ei,0))
        all_pairs = list(itertools.product(cause_idx, effect_idx))
        random.shuffle(all_pairs)
        neg_needed = NEG_RATIO * len(pos)
        negatives = []
        for ci,ei in all_pairs:
            if (ci,ei,0) not in pos and abs(ci-ei)<=PAIR_WINDOW:
                negatives.append((ci,ei,4))
            if len(negatives)>=neg_needed:
                break
        targets = list(pos) + negatives
        if not targets:
            continue
        ci = torch.tensor([t[0] for t in targets], device=device)
        ei = torch.tensor([t[1] for t in targets], device=device)
        lbl= torch.tensor([t[2] for t in targets], device=device)
        logit_sel = logits_rel[b_idx, ci, ei]
        l = ce(logit_sel, lbl)
        l = (l * w_rel[lbl]).mean()
        rel_losses.append(l)
    l3 = torch.stack(rel_losses).mean() if rel_losses else torch.tensor(0.0, device=device)
    return LAMBDAS["cls"]*l1 + LAMBDAS["span"]*ls + LAMBDAS["rel"]*l3

# Training routine
def run_train(model, train_ds, dev_ds, collate):
    print("Starting training...")
    w_span, w_rel = compute_class_weights(train_ds.recs)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    dev_loader   = torch.utils.data.DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    opt = AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(opt, WARMUP_STEPS, EPOCHS*len(train_loader)//GRAD_ACCUM)
    best_mac = 0.0
    step = 0
    for ep in range(EPOCHS):
        print(f"Epoch {ep+1}/{EPOCHS}")
        model.train(); tot=0; n=0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}", leave=False)
        for batch in pbar:
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = loss_fn(batch, out, w_span, w_rel) / GRAD_ACCUM
            loss.backward()
            if (step+1)%GRAD_ACCUM==0:
                clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step(); sched.step(); opt.zero_grad()
            tot+=loss.item(); n+=1; step+=1
            pbar.set_postfix({"loss": f"{tot/n:.4f}"})
        print(f"Epoch {ep} train loss {tot/n:.4f}")
        macro = evaluate_dev(model, dev_loader)
        if macro>best_mac:
            best_mac=macro
            torch.save(model.state_dict(), "gold_joint.ckpt")

def evaluate_dev(model, loader):
    model.eval(); corr=tot=0
    with torch.no_grad():
        for b in loader:
            ids=b["input_ids"].to(device); attn=b["attention_mask"].to(device)
            cls,span,rel=model(ids,attn)
            pred=(span.argmax(-1)>0).any(-1).long()
            corr+=(pred.cpu()==b["cls_labels"]).sum().item(); tot+=len(pred)
    acc=corr/tot; print(f"Dev sentence acc (span‑only) {acc:.3f}"); return acc
