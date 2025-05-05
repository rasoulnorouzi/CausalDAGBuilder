import json
import pathlib
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

ENCODER_NAME = "google-bert/bert-base-uncased"
MAX_LEN = 256

tok = AutoTokenizer.from_pretrained(ENCODER_NAME)

# Data loading helper
def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            text = raw.get("text") or raw.get("content")
            ents = []
            for e in raw.get("labels", []) or raw.get("entities", []) or raw.get("spans", []):
                if isinstance(e, dict):
                    s, e_, lbl = e["start_offset"], e["end_offset"], e["label"].lower()
                else:
                    s, e_, lbl = e
                    lbl = lbl.lower()
                ents.append((int(s), int(e_), lbl))
            rels = []
            for r in raw.get("relations", []) or raw.get("links", []):
                if isinstance(r, dict):
                    src, tgt, pol = r["from_id"], r["to_id"], r.get("type", "none").lower()
                else:
                    src, tgt, pol = r
                rels.append((src, tgt, pol.lower()))
            out.append({"text": text, "entities": sorted(ents, key=lambda x: x[0]), "relations": rels})
    return out

# Dataset
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CausalDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], mode: str):
        self.recs = records
        self.mode = mode

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        rec = self.recs[idx]
        enc = tok(rec["text"], truncation=True, max_length=MAX_LEN,
                   return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        # ----- build BIO labels
        bio = [0] * len(ids)  # 0=O,1=B,2=I (cause) / 3=B,4=I (effect)
        cause_idx, effect_idx = [], []     # span start token indices
        for s, e, l in rec["entities"]:
            first_tok = None
            for i, (ts, te) in enumerate(offsets):
                if ts >= s and te <= e:
                    if first_tok is None:
                        bio[i] = 1 if l == "cause" else 3
                        first_tok = i
                        if l == "cause":
                            cause_idx.append(i)
                        else:
                            effect_idx.append(i)
                    else:
                        bio[i] = 2 if l == "cause" else 4
        return {
            "input_ids": torch.tensor(ids),
            "attn": torch.tensor(enc["attention_mask"]),
            "offsets": offsets,
            "bio": torch.tensor(bio),
            "cause_idx": cause_idx,
            "effect_idx": effect_idx,
            "text": rec["text"],
        }

# Collate function
def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = tok.pad_token_id
    ids, attn, bio = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id)]))
        attn.append(torch.cat([x["attn"], torch.zeros(pad)]))
        bio.append(torch.cat([x["bio"], torch.zeros(pad)]))
    out = {
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(attn),
        "bio_labels": torch.stack(bio).long(),
        "texts": [x["text"] for x in batch],
    }
    out["cls_labels"] = torch.tensor([1 if x["cause_idx"] or x["effect_idx"] else 0 for x in batch])
    out["batch_meta"] = batch
    return out
