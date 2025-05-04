"""
Causal Relation Evaluation Module

This module provides tools for evaluating causal relation extraction systems against gold standard annotations.
It supports evaluation across three tasks:
1. Document-level causal relation detection (Task1)
2. Span-level cause and effect entity detection (Task2)
3. Causal relation extraction between cause-effect pairs (Task3)

Usage:
    results = evaluate(gold_path, pred_path, scenario_task2='A')
    display_results(results, "A")

Input files should be in JSONL format with each line containing document text, entities, and relations.
"""
from __future__ import annotations
import json, pathlib
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
Span = Tuple[int,int]
Entity = Tuple[int,int,str,str]
Relation = Tuple[str,str,str]

def intervals_overlap(a:Span, b:Span) -> bool:
    """
    Check if two span intervals overlap.
    
    Args:
        a: First span tuple (start, end)
        b: Second span tuple (start, end)
        
    Returns:
        True if spans overlap, False otherwise
    """
    return max(a[0],b[0]) < min(a[1],b[1])

def pair_spans(g:List[Span], p:List[Span]):
    """
    Match predicted spans to gold spans based on overlap.
    
    Args:
        g: List of gold standard spans (start, end)
        p: List of predicted spans (start, end)
        
    Returns:
        Tuple of (true positives, false positives, false negatives)
    """
    used=[False]*len(g); tp=0
    for pr in p:
        for i,gr in enumerate(g):
            if not used[i] and intervals_overlap(pr,gr):
                used[i]=True; tp+=1; break
    fp=len(p)-tp; fn=len(g)-tp
    return tp,fp,fn

def _parse_ents(raw, doc_id):
    """
    Parse entities from various input formats into a standardized format.
    
    Args:
        raw: Raw input data containing entities in different formats
        doc_id: Document identifier
        
    Returns:
        List of standardized entity tuples (start, end, label, entity_id)
    """
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
    """
    Parse relations from various input formats into a standardized format.
    
    Args:
        raw: Raw input data containing relations in different formats
        
    Returns:
        List of standardized relation tuples (source_id, target_id, relation_type)
    """
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

def load_jsonl_list(path) -> List[Dict]:
    """
    Load and parse documents from a JSONL file.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        List of parsed documents with text, entities, and relations
    """
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
    """
    Evaluate Task 1: Document-level causal relation detection.
    
    Args:
        gold: List of gold standard documents
        pred: List of predicted documents
        
    Returns:
        Dictionary with precision, recall, F1, and other metrics
    """
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
    """
    Evaluate Task 2: Span-level cause and effect entity detection.
    
    Args:
        gold: List of gold standard documents
        pred: List of predicted documents
        scenario: Evaluation scenario ('A' for all documents, 'B' for only documents with causal relations)
        
    Returns:
        Dictionary with metrics for cause and effect spans
    """
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
    """
    Evaluate Task 3: Causal relation extraction between cause-effect pairs.
    
    Args:
        gold: List of gold standard documents
        pred: List of predicted documents
        
    Returns:
        Dictionary with precision, recall, F1, and other metrics
    """
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

def evaluate(gold_path, pred_path, scenario_task2='A'):
    """
    Main evaluation function that runs all three evaluation tasks.
    
    Args:
        gold_path: Path to the gold standard JSONL file
        pred_path: Path to the predictions JSONL file
        scenario_task2: Evaluation scenario for Task 2 ('A' or 'B')
        
    Returns:
        Dictionary containing results for all three tasks
        
    Raises:
        ValueError: If gold and prediction files don't match in length or content order
    """
    gold=load_jsonl_list(gold_path); pred=load_jsonl_list(pred_path)
    if len(gold)!=len(pred):
        raise ValueError("Length mismatch")
    if gold[0]["text"][:30]!=pred[0]["text"][:30]:
        raise ValueError("Order mismatch")
    return {"Task1":_task1(gold,pred),
            "Task2":_task2(gold,pred,scenario_task2),
            "Task3":_task3(gold,pred)}

def display_results(results, scenario):
    """
    Display evaluation results in a clear, structured format.
    
    Args:
        results: Results dictionary returned by the evaluate function
        scenario: Scenario identifier to display in the header
        
    Prints:
        Formatted evaluation results with task information and metrics
    """
    border = "=" * 70
    title = f" Scenario {scenario} Results "
    
    print(f"\n{border}")
    print(f"{title:^70}")
    print(f"{border}")
    
    if isinstance(results, dict):
        for metric_type, metrics in results.items():
            print(f"\n【 {metric_type} 】")
            
            if isinstance(metrics, dict):
                # Find the longest metric name for alignment
                max_len = max(len(str(key)) for key in metrics.keys())
                separator = "-" * 60
                
                # Table header
                print(f"{separator}")
                print(f"{'Metric':<{max_len+5}} | {'Value':>15}")
                print(f"{separator}")
                
                # Print each metric with proper alignment
                for key, value in metrics.items():
                    metric_name = f"{str(key):<{max_len+5}}"
                    
                    if isinstance(value, float):
                        # Format float values with 4 decimal places
                        print(f"{metric_name} | {value:>15.4f}")
                    elif isinstance(value, dict):
                        # Handle nested dictionary values - print each nested value on its own line
                        print(f"{metric_name} |")
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, float):
                                print(f"  - {nested_key:<{max_len}} | {nested_value:>15.4f}")
                            else:
                                print(f"  - {nested_key:<{max_len}} | {nested_value:>15}")
                    else:
                        print(f"{metric_name} | {value:>15}")
                        
                print(f"{separator}")
            else:
                print(f"  {metrics}")
    else:
        print(f"{results}")
    
    print(f"\n{border}\n")