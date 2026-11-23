"""Create group-audited video manifests for Celeb-DF-v2 or partial DFDC.

The DFDC input is a *subset* of the competition training archives, NOT
the official challenge test partition. Our split names are experimental.
Celeb official test videos retain the published 518-video split.
"""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np

FIELDS=["path","label","split","group_id","video_id","method","official_split"]

def assign(groups,seed=42,ratios=(.6,.15,.125,.125)):
    rng=np.random.default_rng(seed)
    groups=sorted(groups)
    rng.shuffle(groups)
    n=len(groups)
    cuts=[round(n*ratios[0]),round(n*sum(ratios[:2])),round(n*sum(ratios[:3]))]
    return {g:("fit" if i<cuts[0] else "cal" if i<cuts[1] else "val" if i<cuts[2] else "test")
            for i,g in enumerate(groups)}

def celeb(root,seed):
    root=Path(root); testfile=root/"List_of_testing_videos.txt"
    if not testfile.is_file():raise FileNotFoundError(testfile)
    testing=set()
    for line in testfile.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():continue
        parts=line.strip().split()
        rel=parts[-1].replace("\\","/")
        testing.add(rel)
    if len(testing)!=518:
        raise ValueError(f"Expected 518 official Celeb-DF test videos, found {len(testing)}")
    rows=[]
    for dirname,label in (("Celeb-real",0),("YouTube-real",0),("Celeb-synthesis",1)):
        base=root/dirname
        if not base.is_dir():raise FileNotFoundError(base)
        for path in sorted(base.glob("*.mp4")):
            rel=f"{dirname}/{path.name}"
            # Fake idA_idB_#### shares source-video lineage with real idA_####,
            # and may also involve target idB. Use source-video key as a
            # conservative first-level grouping; identity overlap is audited.
            name=path.stem; parts=name.split("_")
            if dirname=="Celeb-synthesis" and len(parts)>=3:
                source=f"{parts[0]}_{parts[-1]}"
            elif dirname=="Celeb-real" and len(parts)>=2:
                source=name
            else:
                source="youtube_"+name
            rows.append(dict(path=str(path.resolve()),label=str(label),
                group_id=source,video_id=f"{dirname}/{name}",method=dirname,
                official_split="test" if rel in testing else "train",
                split=""))
    found={f"{Path(r['path']).parent.name}/{Path(r['path']).name}" for r in rows}
    missing=testing-found
    if missing:raise ValueError(f"Missing {len(missing)} official test videos, examples: {sorted(missing)[:5]}")
    reserved={r["group_id"] for r in rows if r["official_split"]=="test"}
    # Any nonofficial videos that share the same source-video group are
    # excluded from development, not silently moved into the official test.
    dev=[r for r in rows if r["official_split"]!="test" and r["group_id"] not in reserved]
    test=[r for r in rows if r["official_split"]=="test"]
    partitions=assign({r["group_id"] for r in dev},seed,ratios=(.70,.15,.15,0))
    for r in dev:r["split"]=partitions[r["group_id"]]
    for r in test:r["split"]="test"
    excluded=len(rows)-len(dev)-len(test)
    return dev+test,{"official_test_videos":len(test),"excluded_source_overlap":excluded,
                     "identity_disjoint":False,
                     "note":"Celeb source-video group separation only; identities may overlap across splits."}

def dfdc(root,seed):
    root=Path(root)
    dirs=sorted(p for p in root.glob("dfdc_train_part_*") if p.is_dir())
    if not dirs:raise ValueError(f"No dfdc_train_part_* directories found in {root}")
    rows=[];seen=set();incomplete=0
    for folder in dirs:
        mp=folder/"metadata.json"
        if not mp.is_file():raise FileNotFoundError(mp)
        metadata=json.loads(mp.read_text(encoding="utf-8-sig"))
        if isinstance(metadata,list):
            metadata={r["filename"]:r for r in metadata}
        for filename,info in sorted(metadata.items()):
            name=info.get("filename",filename)
            if name in seen:raise ValueError(f"Duplicate filename across parts: {name}")
            seen.add(name)
            path=folder/name
            if not path.is_file():incomplete+=1;continue
            label=info["label"].upper()
            if label not in ("REAL","FAKE"):raise ValueError(f"Unknown label: {label}")
            original=info.get("original")
            group=str(original) if label=="FAKE" and original else str(name)
            rows.append(dict(path=str(path.resolve()),label=str(int(label=="FAKE")),
               group_id=group,video_id=name,method=folder.name,
               official_split="train",split=""))
    # Ensure all connected real+fake videos share a source group
    mapping={r["video_id"]:r for r in rows}
    for r in rows:
        # Fake group references real video's filename even if real not downloaded.
        parent=mapping.get(r["group_id"])
        if parent is not None and parent["label"]!="0":
            raise ValueError("DFDC original link points to a non-real sample")
    part=assign({r["group_id"] for r in rows},seed)
    for r in rows:r["split"]=part[r["group_id"]]
    return rows,{"downloaded_parts":[p.name for p in dirs],
                 "missing_video_files":incomplete,"official_test":False,
                 "note":"Author-defined group-disjoint evaluation on a partial DFDC TRAIN subset."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("celeb","dfdc"),required=True)
    p.add_argument("--root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    rows,notes=(celeb(a.root,a.seed) if a.dataset=="celeb" else dfdc(a.root,a.seed))
    if not rows:raise ValueError("No videos found")
    counts=defaultdict(lambda:defaultdict(int))
    groups=defaultdict(set)
    for r in rows:
        counts[r["split"]][r["label"]]+=1;groups[r["group_id"]].add(r["split"])
    leakage=[g for g,x in groups.items() if len(x)>1]
    if leakage:raise ValueError(f"Source group leakage: {leakage[:5]}")
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    print(json.dumps({"dataset":a.dataset,"videos":len(rows),"source_groups":len(groups),
        "split_real_fake":{s:dict(x) for s,x in counts.items()},**notes},indent=2))

if __name__=="__main__":main()
