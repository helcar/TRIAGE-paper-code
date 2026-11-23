"""Align video manifest frames using the same DeepfakeBench dlib geometry.

Creates frames at <output-root>/<dataset split>/<video index>/*.png and
a compatible frame manifest in video-manifest order. Completion markers
allow resuming a long preprocessing job without losing existing crops.
"""
import argparse,csv,json
from pathlib import Path
import cv2,dlib
from .preprocess_ffpp_dlib import indices,load_bench_alignment

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-manifest",required=True)
    p.add_argument("--output-root",required=True)
    p.add_argument("--frame-manifest",required=True)
    p.add_argument("--bench-dir",required=True)
    p.add_argument("--frames-per-video",type=int,default=8)
    p.add_argument("--selected-per-video",type=int,default=2)
    p.add_argument("--limit-videos",type=int,default=None)
    a=p.parse_args()
    if not 1<=a.selected_per_video<=a.frames_per_video:
        p.error("Require 1 <= selected-per-video <= frames-per-video")
    with open(a.video_manifest,newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError("Empty video manifest")
    root=Path(a.output_root)
    predictor=Path(a.bench_dir)/"preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"
    detector=dlib.get_frontal_face_detector()
    landmark=dlib.shape_predictor(str(predictor))
    align=load_bench_alignment(a.bench_dir)
    result=[];success=0;skipped=0
    if a.limit_videos is not None:rows=rows[:a.limit_videos]
    for pos,r in enumerate(rows):
        # Positional ID is stable if video manifest does not change.
        directory=root/r["split"]/f"{pos:07d}"
        directory.mkdir(parents=True,exist_ok=True)
        marker=directory/"_complete.json"
        if marker.exists():
            saved=[directory/name for name in json.loads(marker.read_text())["files"]]
            if any(not x.is_file() for x in saved):
                raise RuntimeError(f"Corrupted saved crop set at {directory}")
        else:
            cap=cv2.VideoCapture(r["path"])
            if not cap.isOpened():
                skipped+=1;print(f"Unopenable video: {r['path']}",flush=True);continue
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            saved=[]
            for frame_index in indices(total,a.frames_per_video):
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
                ok,image=cap.read()
                if not ok:continue
                face,_,_=align(detector,landmark,image,res=256)
                if face is None:continue
                dst=directory/f"{frame_index:06d}.png"
                if not cv2.imwrite(str(dst),face):raise OSError(dst)
                saved.append(dst)
            cap.release()
            marker.write_text(json.dumps({"source":r["path"],"files":[x.name for x in saved]}))
        if not saved:skipped+=1;continue
        chosen=[saved[i] for i in indices(len(saved),a.selected_per_video)]
        for path in chosen:
            item={k:r[k] for k in ("label","split","group_id","video_id","method","official_split")}
            item["path"]=str(path.resolve())
            result.append(item)
        success+=1
        if (pos+1)%100==0:print(f"Preprocessed {pos+1}/{len(rows)} videos",flush=True)
    if a.limit_videos is not None:
        print("LIMITED preprocessing: manifest is INCOMPLETE and cannot be used for final evaluation.")
    out=Path(a.frame_manifest);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        fields=("path","label","split","group_id","video_id","method","official_split")
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(result)
    from collections import Counter
    counts=Counter(r["split"] for r in result)
    print(json.dumps({"videos_with_faces":success,"videos_without_faces":skipped,
                      "frames":len(result),"by_split":dict(counts),
                      "manifest":str(out)},indent=2))

if __name__=="__main__":main()
