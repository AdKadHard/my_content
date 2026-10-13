"""Независимая эмоциональная разметка фраз двумя HF-моделями + сверка с разметкой Claude v2.
Выход: local/text_emotion/<code>.json, local/text_emotion/agreement.txt"""
import json, collections
from transformers import pipeline
from _common import *
T = ensure(OUT / "text_emotion")
cedr = pipeline("text-classification", model="cointegrated/rubert-tiny2-cedr-emotion-detection", top_k=None)
goem = pipeline("text-classification", model="seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions", top_k=3)
MAP = {"joy": "joy", "sadness": "sadness", "surprise": "surprise", "fear": "fear", "anger": "anger", "no_emotion": "neutral"}
agree = collections.Counter(); n = 0
for code in codes_hits():
    out = T / f"{code}.json"
    sents = sentences(code)
    if not out.exists():
        rows = []
        for s in sents:
            c = {x["label"]: x["score"] for x in cedr(s["text"])[0]}
            g = goem(s["text"])[0]
            rows.append({"i": s["i"], "cedr": c, "cedr_top": MAP.get(max(c, key=c.get), "neutral"), "goemotions_top3": g})
        json.dump(rows, open(out, "w"), ensure_ascii=False, indent=0)
    rows = json.load(open(out)); claude = json.load(open(EMO / "v2" / f"labels_{code}.json"))["sentences"]
    for r, c in zip(rows, claude):
        n += 1; agree[(c["emotion"], r["cedr_top"])] += 1
    print("ok", code)
with open(T / "agreement.txt", "w") as f:
    f.write(f"n={n}\nclaude_v2 -> cedr_top (count)\n")
    for (a, b), k in agree.most_common(): f.write(f"{a:10} {b:10} {k}\n")
