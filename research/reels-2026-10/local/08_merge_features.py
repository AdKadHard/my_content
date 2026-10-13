"""Сборка всех локальных признаков в один CSV на ролик: local/features_local.csv"""
import json, csv, numpy as np
from _common import *
emb = {r["code"]: r for r in csv.DictReader(open(OUT / "embeddings.csv"))} if (OUT / "embeddings.csv").exists() else {}
rows = []
for code in codes_hits():
    r = {"code": code}
    ts = OUT / "timestamps" / f"{code}.json"
    if ts.exists():
        d = json.load(open(ts)); st = {s["i"]: s for s in d["sentences"]}
        lab = json.load(open(EMO / f"labels_{code}.json"))["sentences"]
        fn = {x["i"]: x["function"] for x in lab}
        first = lambda kinds: next((st[i]["start"] for i in sorted(st) if fn.get(i) in kinds and st[i]["start"] is not None), None)
        r.update(t_promise_real=first({"promise"}), t_step_real=first({"step", "demo"}), t_payoff_real=first({"payoff"}),
                 hook_end_s=st[0]["end"] if 0 in st and st[0]["end"] else None, speech_start_s=d["words"][0]["start"] if d["words"] else None)
    pr = OUT / "prosody" / f"{code}.json"
    if pr.exists():
        d = json.load(open(pr)); ps = d["per_sec"]; sent = [x for x in d["per_sentence"] if "wpm" in x]
        r.update(rms_mean=np.mean([x["rms"] for x in ps]), rms_first3=np.mean([x["rms"] for x in ps[:3]]),
                 f0_range_mean=np.nanmean([x["f0_range"] or np.nan for x in ps]), pauses_per_min=d["n_pauses_300ms"] / d["dur"] * 60,
                 wpm_hook=sent[0]["wpm"] if sent else None, wpm_mean=np.mean([x["wpm"] for x in sent]) if sent else None)
    se = OUT / "speech_emotion" / f"{code}.json"
    if se.exists():
        d = [x for x in json.load(open(se)) if "resd" in x]
        if d:
            agg = {}
            for x in d:
                for k, v in x["resd"].items(): agg[k] = agg.get(k, 0) + v / len(d)
            r.update({f"voice_{k}": round(v, 3) for k, v in agg.items()})
            r["voice_hook_top"] = max(d[0]["resd"], key=d[0]["resd"].get)
    vf = OUT / "video_features" / f"{code}.json"
    if vf.exists():
        d = json.load(open(vf)); r.update(n_cuts=d["n_cuts"], cuts_per_10s=d["cuts_per_10s"], first_cut_s=d["first_cut_s"],
                                           cuts_first_5s=d["cuts_first_5s"], ocr_t0=int(bool(d["ocr"]["t0"])), ocr_any=int(any(d["ocr"].values())))
    if code in emb: r.update({k: v for k, v in emb[code].items() if k != "code"})
    rows.append(r)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "code", k))
with open(OUT / "features_local.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("ok", len(rows), "rows,", len(keys), "columns")
