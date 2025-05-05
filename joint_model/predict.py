import torch
import json
from torch.utils.data import DataLoader
from data import collate  # Ensure we use the correct collate function

def spans_from_logits(span_logits, offsets):
    preds=[]
    lab=span_logits.argmax(-1)
    i=0
    while i<len(lab):
        if lab[i]==1:  # B‑cause
            if i >= len(offsets): break
            s=offsets[i][0]; label="cause"; i+=1
            while i<len(lab) and lab[i]==2: i+=1
            if i-1 >= len(offsets): break
            e=offsets[i-1][1]; preds.append((s,e,label))
        elif lab[i]==3:  # B‑effect
            if i >= len(offsets): break
            s=offsets[i][0]; label="effect"; i+=1
            while i<len(lab) and lab[i]==4: i+=1
            if i-1 >= len(offsets): break
            e=offsets[i-1][1]; preds.append((s,e,label))
        else:
            i+=1
    return preds

def predict_to_jsonl(model, test_ds, ckpt, out_path):
    import itertools
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
    loader=DataLoader(test_ds,batch_size=8,shuffle=False,collate_fn=collate)
    entity_id_counter = itertools.count(10000)  # start from a high number to avoid collision
    relation_id_counter = itertools.count(20000)
    example_id_counter = itertools.count(9000)
    with out_path.open("w", encoding="utf-8") as fw, torch.no_grad():
        for batch in loader:
            ids=batch["input_ids"].to(device); attn=batch["attention_mask"].to(device)
            cls,span,rel=model(ids,attn)
            for i, text in enumerate(batch["texts"]):
                spans=spans_from_logits(span[i], batch["batch_meta"][i]["offsets"])
                # Build entities in the same style as doccano
                entities = []
                for s, e, l in spans:
                    entities.append({
                        "id": next(entity_id_counter),
                        "label": l,
                        "start_offset": s,
                        "end_offset": e
                    })
                # No predicted relations, but keep the structure
                relations = []
                # Compose the doccano-style record
                doc = {
                    "id": next(example_id_counter),
                    "text": text,
                    "entities": entities,
                    "relations": relations,
                    "Comments": []
                }
                fw.write(json.dumps(doc)+"\n")
