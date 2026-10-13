"""Видео: склейки (PySceneDetect), кадры на 0/1/2/3 с и в середине, OCR текст-плашек (easyocr).
Выход: local/video_features/<code>.json, local/frames/<code>/t{0,1,2,3,mid}.jpg"""
import json, subprocess
from scenedetect import detect, ContentDetector
import easyocr
from _common import *
VF = ensure(OUT / "video_features"); FR = ensure(OUT / "frames")
reader = easyocr.Reader(["ru", "en"], gpu=False)
for code in codes_hits():
    out = VF / f"{code}.json"; mp4 = OUT / "video" / f"{code}.mp4"
    if out.exists() or not mp4.exists(): continue
    scenes = detect(str(mp4), ContentDetector(threshold=27.0))
    cuts = [s[0].get_seconds() for s in scenes][1:]
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)]))
    d = ensure(FR / code); ocr = {}
    for name, t in [("t0", 0.2), ("t1", 1), ("t2", 2), ("t3", 3), ("mid", dur / 2)]:
        jpg = d / f"{name}.jpg"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(mp4), "-frames:v", "1", "-vf", "scale=540:-1", str(jpg)])
        ocr[name] = [x[1] for x in reader.readtext(str(jpg)) if x[2] > 0.5] if jpg.exists() else []
    json.dump({"dur": dur, "n_cuts": len(cuts), "cuts_per_10s": len(cuts) / dur * 10, "first_cut_s": cuts[0] if cuts else None,
               "cuts_first_5s": sum(c < 5 for c in cuts), "ocr": ocr}, open(out, "w"), ensure_ascii=False, indent=0)
    print("ok", code, len(cuts), "cuts")
