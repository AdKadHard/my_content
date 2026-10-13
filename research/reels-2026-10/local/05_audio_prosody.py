"""Просодия: громкость (RMS), высота (pyin), паузы, темп — по секундам и по фразам (нужен шаг 02).
Выход: local/prosody/<code>.json"""
import json, numpy as np, librosa
import sys
from _common import *
P = ensure(OUT / "prosody")
for code in (codes_all() if "--all" in sys.argv else codes_hits()):
    out = P / f"{code}.json"
    if out.exists(): continue
    wav = OUT / "audio" / f"{code}.wav"
    if not wav.exists(): continue
    y, sr = librosa.load(str(wav), sr=16000)
    hop = 160  # 10 мс
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, hop_length=hop)
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    per_sec = []
    for s in range(int(t[-1]) + 1):
        m = (t >= s) & (t < s + 1)
        fv = f0[m][voiced[m]] if m.any() else np.array([])
        per_sec.append({"t": s, "rms": float(rms[m].mean()) if m.any() else 0, "f0_med": float(np.median(fv)) if len(fv) else None,
                        "f0_range": float(np.ptp(fv)) if len(fv) > 1 else None, "voiced_share": float(voiced[m].mean()) if m.any() else 0})
    ts = OUT / "timestamps" / f"{code}.json"; per_sent = []
    if ts.exists():
        d = json.load(open(ts)); words = d["words"]
        for s in d["sentences"]:
            if s["start"] is None: per_sent.append({"i": s["i"]}); continue
            m = (t >= s["start"]) & (t < s["end"]); fv = f0[m][voiced[m]] if m.any() else np.array([])
            nw = sum(1 for w in words if s["start"] <= w["start"] < s["end"])
            per_sent.append({"i": s["i"], "dur": s["end"] - s["start"], "wpm": nw / max(s["end"] - s["start"], 0.1) * 60,
                             "rms": float(rms[m].mean()) if m.any() else 0, "f0_med": float(np.median(fv)) if len(fv) else None,
                             "f0_range": float(np.ptp(fv)) if len(fv) > 1 else None})
    sil = (rms < np.percentile(rms, 15)); pauses = []; run = 0
    for v in sil:
        run = run + 1 if v else 0
        if run == 30: pauses.append(1)  # пауза ≥ 300 мс
    json.dump({"per_sec": per_sec, "per_sentence": per_sent, "n_pauses_300ms": len(pauses), "dur": float(t[-1])}, open(out, "w"), indent=0)
    print("ok", code)
