"""Эмоция по голосу на каждую фразу (нужен шаг 02): Aniemore wav2vec2 (RESD) и HuBERT (Dusha).
Выход: local/speech_emotion/<code>.json"""
import json, torch, numpy as np, librosa
from transformers import pipeline
from _common import *
S = ensure(OUT / "speech_emotion")
m1 = pipeline("audio-classification", model="Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition", top_k=None)
m2 = pipeline("audio-classification", model="xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned", top_k=None)
for code in codes_hits():
    out = S / f"{code}.json"; ts = OUT / "timestamps" / f"{code}.json"; wav = OUT / "audio" / f"{code}.wav"
    if out.exists() or not ts.exists() or not wav.exists(): continue
    y, sr = librosa.load(str(wav), sr=16000); rows = []
    for s in json.load(open(ts))["sentences"]:
        if s["start"] is None: rows.append({"i": s["i"]}); continue
        a = y[int(s["start"] * sr):int(s["end"] * sr)]
        if len(a) < sr // 2: rows.append({"i": s["i"]}); continue
        r1 = {x["label"]: round(x["score"], 3) for x in m1({"raw": a, "sampling_rate": sr})}
        r2 = {x["label"]: round(x["score"], 3) for x in m2({"raw": a, "sampling_rate": sr})}
        rows.append({"i": s["i"], "resd": r1, "dusha": r2})
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=0); print("ok", code)
