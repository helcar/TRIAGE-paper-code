"""Fine-tune CommunityForensics ViT-Small on FF++ TRIAGE FIT groups only.

Detector training/early-stopping groups are disjoint from one another and
from TRIAGE CAL/VAL/TEST. Detector training overlaps TRIAGE FIT by design;
this is a scoped architecture experiment, not a separately trained backbone.
Never selects checkpoints on TRIAGE VAL/TEST.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .extract_ffpp import load_manifest
from .extract_community_vit_b7 import load_detector, sha256


class Frames(Dataset):
    def __init__(self, rows, root, processor):
        self.rows=rows
        self.root=Path(root)
        self.processor=processor
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        p=Path(r["path"])
        if not p.is_absolute():
            p=self.root/p
        with Image.open(p) as im:
            pixels=self.processor(images=im.convert("RGB"),return_tensors="pt")["pixel_values"][0]
        return pixels,torch.tensor(int(r["label"]),dtype=torch.float32)


def partition(rows, seed, fraction):
    grouped=defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    groups=sorted(grouped)
    rng=random.Random(seed)
    rng.shuffle(groups)
    num=max(1,round(len(groups)*fraction))
    if num>=len(groups):
        raise ValueError("No groups remain for detector fitting")
    held=set(groups[:num])
    train=[r for r in rows if r["group_id"] not in held]
    valid=[r for r in rows if r["group_id"] in held]
    for name,split in (("train",train),("detector_holdout",valid)):
        if {int(r["label"]) for r in split}!={0,1}:
            raise RuntimeError(f"{name} missing class after group partition")
    return train,valid,sorted(held)


@torch.no_grad()
def validate(model,loader,device):
    model.eval()
    ys=[];zs=[]
    for x,y in loader:
        z=model(pixel_values=x.to(device)).logits[:,0]
        zs.extend(z.cpu().tolist())
        ys.extend(y.tolist())
    from sklearn.metrics import accuracy_score
    arr=np.asarray(zs); y=np.asarray(ys).astype(int)
    if len(np.unique(y))!=2:
        raise RuntimeError("Detector holdout lacks both classes")
    return {"auroc":float(roc_auc_score(y,arr)),
            "accuracy":float(accuracy_score(y,arr>=0)),
            "predicted_fake":int(np.count_nonzero(arr>=0)),
            "n":len(y),"real":int(np.count_nonzero(y==0)),
            "fake":int(np.count_nonzero(y==1))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--image-root",default=".")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--revision",default=None)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--holdout-fraction",type=float,default=0.15)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--accumulation-steps",type=int,default=4)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--weight-decay",type=float,default=0.01)
    p.add_argument("--device",default="cuda")
    p.add_argument("--num-workers",type=int,default=0)
    a=p.parse_args()
    if not (0<a.holdout_fraction<0.5) or min(a.epochs,a.batch_size,a.accumulation_steps)<1:
        p.error("Invalid holdout fraction or counts")
    if a.lr<=0:
        p.error("Learning rate must be positive")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    output=Path(a.output_dir)
    if output.exists() and any(output.iterdir()):
        # The previous failed export may leave a header-only manifest.
        # Never silently replace training reports or model checkpoints.
        existing=list(output.iterdir())
        allowed={"detector_train.csv","detector_holdout.csv"}
        if not all(x.is_file() and x.name in allowed and x.stat().st_size < 500 for x in existing):
            p.error("Output directory contains prior results; use a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    manifest=Path(a.manifest)
    all_rows=load_manifest(manifest)
    fit=[r for r in all_rows if r["split"]=="fit"]
    train,valid,held=partition(fit,a.seed,a.holdout_fraction)
    # Freeze the group partition before training and preserve the actual rows.
    for name,rows in (("detector_train",train),("detector_holdout",valid)):
        with (output/(name+".csv")).open("w",encoding="utf-8",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(all_rows[0].keys()))
            writer.writeheader();writer.writerows(rows)
    detector,processor,origin=load_detector(a.revision,a.device)
    model=detector.classifier
    for param in model.parameters():
        param.requires_grad_(True)
    ds=Frames(train,a.image_root,processor)
    labels=np.asarray([int(r["label"]) for r in train])
    counts=np.bincount(labels,minlength=2)
    weights=np.asarray([1.0/counts[y] for y in labels],dtype=float)
    sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),
                                  num_samples=len(train),replacement=True)
    loader=DataLoader(ds,batch_size=a.batch_size,sampler=sampler,
                      num_workers=a.num_workers,pin_memory=a.device.startswith("cuda"))
    vloader=DataLoader(Frames(valid,a.image_root,processor),
                       batch_size=a.batch_size,shuffle=False,
                       num_workers=a.num_workers,pin_memory=a.device.startswith("cuda"))
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    criterion=torch.nn.BCEWithLogitsLoss()
    best_auc=float("-inf")
    history=[]
    for epoch in range(1,a.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum=0.0
        for step,(x,y) in enumerate(loader,1):
            x=x.to(a.device,non_blocking=True);y=y.to(a.device,non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,
                                enabled=a.device.startswith("cuda")):
                z=model(pixel_values=x).logits[:,0]
                loss=criterion(z,y)
            # Scaler needed for stable float16 gradient training.
            if step==1 and epoch==1:
                scaler=torch.amp.GradScaler("cuda",enabled=a.device.startswith("cuda"))
            scaler.scale(loss/a.accumulation_steps).backward()
            if step%a.accumulation_steps==0 or step==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum+=float(loss.detach())*len(y)
        stats=validate(model,vloader,a.device)
        stats.update(epoch=epoch,train_loss=loss_sum/len(train))
        history.append(stats)
        print(json.dumps(stats),flush=True)
        if stats["auroc"]>best_auc:
            best_auc=stats["auroc"]
            torch.save({"state_dict":{k:v.cpu() for k,v in model.state_dict().items()},
                        "epoch":epoch,"metrics":stats,
                        "base_checkpoint_sha256":sha256(origin),
                        "manifest_sha256":sha256(manifest),
                        "seed":a.seed,"processor":processor.to_dict()},
                       output/"best_vit_ffpp.pth")
    summary={"source_checkpoint":str(origin),
             "base_checkpoint_sha256":sha256(origin),
             "manifest_sha256":sha256(manifest),
             "train_images":len(train),"holdout_images":len(valid),
             "train_groups":len({r["group_id"] for r in train}),
             "holdout_groups":len(held),
             "training_labels":dict(Counter(r["label"] for r in train)),
             "holdout_labels":dict(Counter(r["label"] for r in valid)),
             "best_auroc":best_auc,
             "best_epoch":max(history,key=lambda x:x["auroc"])["epoch"],
             "history":history,
             "protocol":"Fine-tune only TRIAGE FIT groups; CAL/VAL/TEST never used for detector selection"}
    (output/"training_report.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("BEST:",json.dumps({"epoch":summary["best_epoch"],"auroc":best_auc,
                               "checkpoint":str(output/"best_vit_ffpp.pth")}),flush=True)


if __name__=="__main__":
    main()
