"""Морфология транскриптов рилзов vs просмотры.
python3 morphology.py transcripts_full_YYYY-MM-DD.json labels_YYYY-MM-DD.csv"""
import csv, json, re, sys, statistics as st
import pymorphy3
m = pymorphy3.MorphAnalyzer()
T = json.load(open(sys.argv[1]))
L = {r["shortcode"]: r for r in csv.DictReader(open(sys.argv[2]))}

def sents(t): return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
def feats(text, dur=None):
    words = re.findall(r"[А-Яа-яЁёA-Za-z]+|\d+[\d.,]*", text)
    n = len(words) or 1
    c = dict(noun=0, verb=0, adj=0, num=0, latin=0, p1s=0, p1p=0, p2=0, imper=0, past=0, pres=0, fut=0, cond=0)
    for w in words:
        if re.match(r"\d", w): c["num"] += 1; continue
        if re.match(r"[A-Za-z]", w): c["latin"] += 1; continue
        lw = w.lower()
        if lw == "бы": c["cond"] += 1
        if lw in ("я", "мне", "меня", "мной", "мой", "моя", "моё", "мое", "мои", "моего", "моем", "моих"): c["p1s"] += 1
        if lw in ("мы", "нам", "нас", "наш", "наша", "наши"): c["p1p"] += 1
        if lw in ("ты", "тебе", "тебя", "тобой", "твой", "твоя", "твое", "твои", "твоих", "твоей", "твоего"): c["p2"] += 1
        p = m.parse(lw)[0]; t = p.tag
        if t.POS == "NOUN": c["noun"] += 1
        if t.POS in ("ADJF", "ADJS", "PRTF", "PRTS"): c["adj"] += 1
        if t.POS == "NUMR": c["num"] += 1
        if t.POS in ("VERB", "INFN"):
            c["verb"] += 1
            if t.mood == "impr": c["imper"] += 1
            if t.tense == "past": c["past"] += 1
            if t.tense == "pres": c["pres"] += 1
            if t.tense == "futr": c["fut"] += 1
            if t.person == "2per": c["p2"] += 1
    S = sents(text)
    f = {k: v / n * 100 for k, v in c.items()}
    f["words"] = n
    f["sent_len"] = n / max(len(S), 1)
    f["questions"] = text.count("?")
    if dur: f["wpm"] = n / dur * 60
    return f

rows = []
for code, text in T.items():
    if code not in L or len(text) < 80: continue  # мемы без речи не считаем
    r = L[code]; v = int(r["views"]); d = float(r["dur"])
    S = sents(text)
    hook = " ".join(S[:2])
    rows.append(dict(code=code, views=v, body=feats(text, d), hook=feats(hook)))

def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0]*len(xs)
    for k, i in enumerate(o): r[i] = k
    return r
def spearman(a, b):
    ra, rb = rank(a), rank(b); return st.correlation(ra, rb)

views = [r["views"] for r in rows]
rows_sorted = sorted(rows, key=lambda r: -r["views"]); third = len(rows)//3
top, bot = rows_sorted[:third], rows_sorted[-third:]
print(f"n={len(rows)} (без мемов), топ/низ по {third}\n")
for part in ("hook", "body"):
    print(f"## {part}: признак | spearman с views | медиана топ-трети | медиана низ-трети")
    keys = rows[0][part].keys()
    res = []
    for k in keys:
        xs = [r[part][k] for r in rows]
        rho = spearman(xs, views)
        res.append((abs(rho), k, rho, st.median(r[part][k] for r in top), st.median(r[part][k] for r in bot)))
    for _, k, rho, a, b in sorted(res, reverse=True):
        print(f"{k:10} rho={rho:+.2f}  top={a:6.1f}  bottom={b:6.1f}")
    print()
