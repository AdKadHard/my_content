"""Сборка признаков ролика: эмоции (v2), структура (функции фраз), контрольные факторы.
Пишет emotion/features_v2.csv. python3 reel_features.py"""
from pathlib import Path as _P
_R = _P(__file__).resolve().parent.parent
D, A, RES = _R / "data", _R / "annotations", _R / "results"
import csv, json, glob, re, math, statistics as st
L = {r["shortcode"]: r for r in csv.DictReader(open(D / "labels_2026-10-12.csv"))}
S = {r["shortcode"]: r for r in csv.DictReader(open(D / "reels_stats_2026-10-12.csv"))}
T = json.load(open(D / "transcripts_full_2026-10-12.json"))
CODE = re.compile(r"(пиши|напиши|пишите|напишешь|напишите)(?:(?![.!?] ).){0,60}(коммент|личк)|в коммент\w*(?:(?![.!?] ).){0,30}(пришл|вышл|скину|отправ)", re.I)
rows = []
for f in sorted(glob.glob(str(A / "labels_*.json"))):
    d = json.load(open(f)); c = d["code"]; s = d["sentences"]; n = len(s)
    lab, st_ = L[c], S[c]
    views = int(lab["views"]); dur = float(lab["dur"])
    sents = json.load(open(A / f"sentences_{c}.json"))["sentences"]
    words = [len(x["text"].split()) for x in sents]; tw = sum(words) or 1
    cum = [sum(words[:k]) / tw * dur for k in range(n)]  # оценка секунды начала фразы
    fn = [x["function"] for x in s]
    def first_t(kinds):
        for k, x in enumerate(fn):
            if x in kinds: return cum[k]
        return dur
    seq = [fn[0]] + [fn[k] for k in range(1, n) if fn[k] != fn[k - 1]]
    p_idx = next((k for k, x in enumerate(fn) if x == "promise"), None)
    pr_idx = next((k for k, x in enumerate(fn) if x in ("proof", "demo")), None)
    hook_mech = s[0].get("mechanism") or ""
    rows.append(dict(
        code=c, views=views, log_views=math.log10(views),
        comment_rate=int(st_["comments"]) / views * 100, like_rate=int(st_["likes"]) / views * 100,
        topic=lab["topic"], period=lab["period"], dur=dur, hook_type=lab["hook"],
        cta_code=int(bool(CODE.search(T[c] + " " + st_["caption_head"]))),
        enemy=int(any(x.get("mechanism") == "enemy" for x in s)),
        hook_identity=int(hook_mech == "identity"), hook_enemy=int(hook_mech == "enemy"), hook_contrast=int(hook_mech == "contrast"),
        concrete_share=sum(x["concrete"] for x in s) / n, hook_concrete=int(any(x["concrete"] for x in s[:2])),
        open_loops=sum(x["loop"] == "open" for x in s),
        neg_viewer_share=sum(x["emotion"] in ("anxiety", "anger", "fear", "sadness", "disgust") for x in s) / n,
        t_promise=first_t({"promise"}), t_proof=first_t({"proof", "demo"}), t_step=first_t({"step", "demo"}),
        t_payoff=first_t({"payoff"}),
        promise_before_proof=int(p_idx is not None and (pr_idx is None or p_idx < pr_idx)),
        has_promise=int("promise" in fn), has_payoff=int("payoff" in fn), has_twist=int("twist" in fn),
        n_blocks=len(seq), blocks_per_10s=len(seq) / dur * 10,
        step_share=sum(x in ("step", "demo") for x in fn) / n, insight_share=sum(x == "insight" for x in fn) / n,
        seq=">".join(seq[:6]), second=fn[1] if n > 1 else "",
    ))
keys = list(rows[0])
with open(RES / "features_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(len(rows), "роликов;", "с кодом в комменты:", sum(r["cta_code"] for r in rows))
