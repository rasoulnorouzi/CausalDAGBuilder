import torch
from torch import nn
from transformers import AutoModel

ENCODER_NAME = "google-bert/bert-base-uncased"

class JointCausalIE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(ENCODER_NAME)
        hid = self.enc.config.hidden_size
        self.cls_head = nn.Linear(hid, 2)
        self.span_head = nn.Linear(hid, 5)   # O/B‑C/I‑C/B‑E/I‑E
        self.biaff_start = nn.Linear(hid, hid)
        self.biaff_end   = nn.Linear(hid, hid)
        self.rel_out     = nn.Linear(hid*2, 5)  # 4 polarities + none
    def forward(self, ids, attn):
        h = self.enc(ids, attention_mask=attn).last_hidden_state
        logits_cls  = self.cls_head(h[:,0])          # [B,2]
        logits_span = self.span_head(h)              # [B,T,5]
        qs = self.biaff_start(h)                     # [B,T,H]
        ke = self.biaff_end(h)                       # [B,T,H]
        pair_vec = torch.cat([qs.unsqueeze(2).expand(-1,-1,qs.size(1),-1),
                              ke.unsqueeze(1).expand(-1,ke.size(1),-1,-1)], dim=-1)
        logits_rel = self.rel_out(pair_vec)          # [B,T,T,5]
        return logits_cls, logits_span, logits_rel
