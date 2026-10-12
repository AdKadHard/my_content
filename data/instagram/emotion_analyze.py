"""Сводка психоэмоциональной разметки vs метрики.
python3 emotion_analyze.py labels_YYYY-MM-DD.csv reels_stats_YYYY-MM-DD.csv"""
import csv, glob, json, sys, statistics as st
from collections import Counter
L = {r["shortcode"]: r for r in csv.DictReader(open(sys.argv[1]))}
S = {r["shortcode"]: r for r in csv.DictReader(open(sys.argv[2]))}

def arc_shape(v):
    n = len(v)
    if n < 4: return "short"
    a, b, c = v[: n // 3], v[n // 3 : 2 * n // 3], v[2 * n // 3 :]
    ma, mb, mc = st.mean(a), st.mean(b), st.mean(c)
    d1, d2 = mb - ma, mc - mb
    th = 0.25
    if d1 > th and d2 > th: return "rise"
    if d1 < -th and d2 < -th: return "fall"
    if d1 < -th and d2 > th: return "fall-rise"
    if d1 > th and d2 < -th: return "rise-fall"
    if abs(mc - ma) <= th: return "flat"
    return "rise" if mc > ma else "fall"

rows = []
for f in sorted(glob.glob("emotion/labels_*.json")):
    d = json.load(open(f)); c = d["code"]; s = d["sentences"]
    if c not in L: continue
    views = int(L[c]["views"]); likes = int(S[c]["likes"]); com = int(S[c]["comments"])
    val = [x["valence"] for x in s]; aro = [x["arousal"] for x in s]
    hook = s[:2]
    opens = [x["i"] for x in s if x["loop"] == "open"]
    closes = [x["i"] for x in s if x["loop"] == "close"]
    mech = Counter(x["mechanism"] for x in s if x.get("mechanism"))
    emo = Counter(x["emotion"] for x in s)
    n = len(s)
    rows.append(dict(
        code=c, views=views, topic=L[c]["topic"],
        like_rate=likes / views * 100, comment_rate=com / views * 100,
        hook_arousal=st.mean(x["arousal"] for x in hook),
        hook_valence=st.mean(x["valence"] for x in hook),
        hook_open_loop=int(any(x["loop"] == "open" for x in hook)),
        hook_concrete=int(any(x["concrete"] for x in hook)),
        hook_you=int(any(x["you_focus"] for x in hook)),
        mean_arousal=st.mean(aro), max_arousal=max(aro), arousal_sd=st.pstdev(aro),
        mean_valence=st.mean(val), neg_share=sum(v < 0 for v in val) / n,
        open_loops=len(opens), loops_closed=min(len(closes), len(opens)) / len(opens) if opens else None,
        first_payoff=next((x["i"] / n for x in s if x["function"] in ("payoff", "demo", "proof")), 1.0),
        you_share=sum(x["you_focus"] for x in s) / n, concrete_share=sum(x["concrete"] for x in s) / n,
        curiosity_share=emo["curiosity"] / n, anxiety_anger_share=(emo["anxiety"] + emo["anger"] + emo["fear"]) / n,
        awe_surprise_share=(emo["awe"] + emo["surprise"]) / n, trust_share=emo["trust"] / n,
        enemy=int(mech["enemy"] > 0), self_disclosure=int(mech["self_disclosure"] > 0),
        contrast=int(mech["contrast"] > 0), loss_aversion=int(mech["loss_aversion"] > 0),
        reciprocity=int(mech["reciprocity"] > 0), authority=int(mech["authority"] > 0),
        arc=arc_shape(val), hook_emotion=hook[0]["emotion"], hook_mech=hook[0].get("mechanism") or "-",
    ))

def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def rho(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None]
    if len(pairs) < 8: return float("nan")
    x, y = zip(*pairs); return st.correlation(rank(list(x)), rank(list(y)))

k = lambda v: f"{v/1000:.0f}K"
print(f"n={len(rows)} роликов\n")
num = [f for f in rows[0] if isinstance(rows[0][f], (int, float)) and f not in ("views", "like_rate", "comment_rate")]
print("## Признак | rho с views | rho с comments/views | rho с likes/views")
res = []
for f in num:
    xs = [r[f] for r in rows]
    res.append((abs(rho(xs, [r['views'] for r in rows])), f, rho(xs, [r['views'] for r in rows]), rho(xs, [r['comment_rate'] for r in rows]), rho(xs, [r['like_rate'] for r in rows])))
for _, f, a, b, c in sorted(res, reverse=True):
    print(f"{f:20} views {a:+.2f}   comments {b:+.2f}   likes {c:+.2f}")
for cat in ("arc", "hook_emotion", "hook_mech"):
    print(f"\n## {cat}")
    g = {}
    for r in rows: g.setdefault(r[cat], []).append(r["views"])
    for key, v in sorted(g.items(), key=lambda x: -st.median(x[1])):
        print(f"{key:16} n={len(v):2}  median={k(st.median(v)):>6}")
json.dump(rows, open("emotion/summary.json", "w"), ensure_ascii=False, indent=1)
