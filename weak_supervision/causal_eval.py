from __future__ import annotations
import json, pathlib
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
Span = Tuple[int,int]
Entity = Tuple[int,int,str,str]
Relation = Tuple[str,str,str]

def intervals_overlap(a:Span,b:Span)->bool:
    return max(a[0],b[0])<min(a[1],b[1])

def pair_spans(g:List[Span], p:List[Span]):
    used=[False]*len(g); tp=0
    for pr in p:
        for i,gr in enumerate(g):
            if not used[i] and intervals_overlap(pr,gr):
                used[i]=True; tp+=1; break
    fp=len(p)-tp; fn=len(g)-tp
    return tp,fp,fn

def _parse_ents(raw, doc_id):
    out=[]
    for ent in raw.get("labels",[]) or raw.get("entities",[]) or raw.get("spans",[]):
        if isinstance(ent,dict):
            s=ent.get("start_offset") or ent.get("start"); e=ent.get("end_offset") or ent.get("end")
            lbl=ent.get("label") or ent.get("type") or ""
            eid=str(ent.get("id") or ent.get("entity_id") or f"{doc_id}:{s}-{e}")
        else:
            if len(ent)==3: s,e,lbl=ent
            elif len(ent)==4: _,s,e,lbl=ent
            else: continue
            eid=f"{doc_id}:{s}-{e}"
        if s is None or e is None: continue
        out.append((int(s),int(e),str(lbl).lower(),eid))
    return out

def _parse_rels(raw):
    out=[]
    for rel in raw.get("relations",[]) or raw.get("links",[]):
        if isinstance(rel,dict):
            src=str(rel.get("from_id") or rel.get("src")); tgt=str(rel.get("to_id") or rel.get("dst"))
            pol=str(rel.get("type") or rel.get("label") or rel.get("relation")).lower()
        else:
            if len(rel)<3: continue
            src,tgt,pol=rel[:3]; pol=str(pol).lower()
        out.append((src,tgt,pol))
    return out

def load_jsonl_list(path)->List[Dict]:
    docs=[]
    with pathlib.Path(path).open("r",encoding="utf-8") as fh:
        for line in fh:
            if not line.strip(): continue
            raw=json.loads(line)
            did=str(raw.get("id") or raw.get("pk") or raw.get("doc_id") or raw.get("document_id") or len(docs))
            docs.append({"text":raw.get("text") or raw.get("content") or "", 
                         "entities":_parse_ents(raw,did),
                         "relations":_parse_rels(raw)})
    return docs

def _task1(gold, pred):
    tp=fp=fn=tn=0
    for g,p in zip(gold,pred):
        gc=any(l in {"cause","effect"} for *_,l,_ in g["entities"])
        pc=any(l in {"cause","effect"} for *_,l,_ in p["entities"])
        if gc and pc: tp+=1
        elif gc and not pc: fn+=1
        elif not gc and pc: fp+=1
        else: tn+=1
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    f1=2*tp/(2*tp+fp+fn) if tp else 0; acc=(tp+tn)/len(gold)
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,"Precision":prec,"Recall":rec,"F1":f1,"Accuracy":acc,"N":len(gold)}

def _task2(gold,pred,scenario="A"):
    counts={lbl:Counter() for lbl in ("cause","effect")}
    for g,p in zip(gold,pred):
        gb=defaultdict(list); pb=defaultdict(list)
        for s,e,l,_ in g["entities"]:
            if l in {"cause","effect"}: gb[l].append((s,e))
        for s,e,l,_ in p["entities"]:
            if l in {"cause","effect"}: pb[l].append((s,e))
        gc=bool(gb["cause"] or gb["effect"]); pc=bool(pb["cause"] or pb["effect"])
        if scenario.upper()=="B" and not (gc and pc): continue
        for l in ("cause","effect"):
            tp,fp,fn=pair_spans(gb[l],pb[l]); c=counts[l]
            c["TP"]+=tp; c["FP"]+=fp; c["FN"]+=fn
    rep={}
    for l,c in counts.items():
        tp,fp,fn=c["TP"],c["FP"],c["FN"]
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        f1=2*tp/(2*tp+fp+fn) if tp else 0
        rep[l]={"TP":tp,"FP":fp,"FN":fn,"Precision":prec,"Recall":rec,"F1":f1}
    return rep

def _task3(gold,pred):
    cnt=Counter()
    for g,p in zip(gold,pred):
        ge={eid:(s,e,l) for s,e,l,eid in g["entities"]}
        pe={eid:(s,e,l) for s,e,l,eid in p["entities"]}
        def get(emap,rels):
            res=set()
            for src,tgt,pol in rels:
                if src in emap and tgt in emap:
                    s1,e1,l1=emap[src]; s2,e2,l2=emap[tgt]
                    if l1=="cause" and l2=="effect":
                        res.add(((s1,e1),(s2,e2),pol))
            return res
        gr=get(ge,g["relations"]); pr=get(pe,p["relations"])
        matched=set()
        for pr in pr:
            ps_c,ps_e,pp=pr; hit=False
            for grr in gr:
                gs_c,gs_e,gp=grr
                if gp==pp and intervals_overlap(ps_c,gs_c) and intervals_overlap(ps_e,gs_e) and grr not in matched:
                    matched.add(grr); cnt["TP"]+=1; hit=True; break
            if not hit: cnt["FP"]+=1
        cnt["FN"]+=len(gr)-len(matched)
    tp,fp,fn=cnt["TP"],cnt["FP"],cnt["FN"]
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    f1=2*tp/(2*tp+fp+fn) if tp else 0
    return {"TP":tp,"FP":fp,"FN":fn,"Precision":prec,"Recall":rec,"F1":f1}

def evaluate(gold_path,pred_path,scenario_task2='A'):
    gold=load_jsonl_list(gold_path); pred=load_jsonl_list(pred_path)
    if len(gold)!=len(pred):
        raise ValueError("Length mismatch")
    if gold[0]["text"][:30]!=pred[0]["text"][:30]:
        raise ValueError("Order mismatch")
    return {"Task1":_task1(gold,pred),
            "Task2":_task2(gold,pred,scenario_task2),
            "Task3":_task3(gold,pred)}