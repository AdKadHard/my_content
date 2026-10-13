"""Сторителлинг-разметка v4 (template/conflict/stakes/pov после калибровки с Аделем; остальной блок story из v3)
vs метрики, с контролем темы/периода/длительности/кода. python3 story_analyze_v4.py → results/story_v4_<дата>.txt"""
from pathlib import Path as _P
_R = _P(__file__).resolve().parent.parent
D, A, RES = _R / "data", _R / "annotations", _R / "results"
import csv, json, glob, os, statistics as st
EXCLUDE = set(filter(None, os.environ.get("REELS_EXCLUDE", "").split(",")))
from collections import Counter, defaultdict
import numpy as np
rng = np.random.default_rng(3)
L = {r["shortcode"]: r for r in csv.DictReader(open(D / "labels_2026-10-12.csv"))}
S = {r["shortcode"]: r for r in csv.DictReader(open(D / "reels_stats_2026-10-12.csv"))}
F = {r["code"]: r for r in csv.DictReader(open(RES / "features_v2.csv"))}
rows = []
for f in sorted(glob.glob(str(A / "labels_*.json"))):
    d = json.load(open(f)); c = d["code"]
    if c in EXCLUDE: continue
    s = d["sentences"]; n = len(s); st_ = d["story"]
    v4 = json.load(open(A / "story_v4" / f"{c}.json"))["story"]
    sents = json.load(open(A / f"sentences_{c}.json"))["sentences"]
    words = [len(x["text"].split()) for x in sents]; tw = sum(words); dur = float(L[c]["dur"])
    pos = lambda i: None if i is None else sum(words[:i]) / tw  # доля ролика до фразы
    fr = st_["freytag"]; fn = [x["function"] for x in s]
    rows.append(dict(code=c, views=int(L[c]["views"]), topic=L[c]["topic"], period=L[c]["period"], dur=dur,
        cta_code=int(F[c]["cta_code"]), comments=int(S[c]["comments"]), likes=int(S[c]["likes"]),
        template=v4["template"], conflict=v4["conflict"], stakes="+".join(sorted(v4["stakes"])) or "none", pov=v4["pov"],
        st_money=int("money" in v4["stakes"]), st_time=int("time" in v4["stakes"]), st_knowledge=int("knowledge" in v4["stakes"]),
        st_job=int("job" in v4["stakes"]), st_none=int(not v4["stakes"]), n_stakes=len(v4["stakes"]),
        in_medias_res=st_["in_medias_res"], transformation=st_["transformation"], cliff=st_["cliffhanger_end"],
        circle=set(st_["circle"]), n_circle=len(st_["circle"]),
        climax_pos=pos(fr.get("climax")), has_climax=int(fr.get("climax") is not None),
        has_resolution=int(fr.get("resolution") is not None), has_rising=int(fr.get("rising") is not None),
        twist_share=fn.count("twist") / n, setup_share=fn.count("setup") / n, step_share=sum(x in ("step","demo") for x in fn) / n,
        insight_share=fn.count("insight") / n, has_promise=int("promise" in fn), promise_pos=pos(next((k for k, x in enumerate(fn) if x == "promise"), None)),
        second=fn[1] if n > 1 else "", open_loops=sum(x["loop"] == "open" for x in s)))
k = lambda v: f"{v/1000:.0f}K"
print(f"n={len(rows)}\n")
for cat in ("template", "conflict", "stakes", "pov", "second"):
    print(f"## {cat}: n / медиана views / медиана комментов на 1K просмотров")
    g = defaultdict(list)
    for r in rows: g[r[cat]].append(r)
    for key, v in sorted(g.items(), key=lambda x: -st.median(r["views"] for r in x[1])):
        print(f"{key:16} n={len(v):2}  views={k(st.median(r['views'] for r in v)):>6}  comm/1K={st.median(r['comments']/r['views']*1000 for r in v):5.1f}")
    print()
print("## Круг Хармона: шаг есть / нет → медиана views")
for step in ("need", "go", "search", "find", "take", "return", "change"):
    a = [r["views"] for r in rows if step in r["circle"]]; b = [r["views"] for r in rows if step not in r["circle"]]
    print(f"{step:7} есть n={len(a):2} {k(st.median(a)) if a else '-':>6}   нет n={len(b):2} {k(st.median(b)) if b else '-':>6}")
print("\n## Бинарные: есть / нет → медиана views")
for fld in ("in_medias_res", "transformation", "cliff", "has_climax", "has_resolution", "has_promise"):
    a = [r["views"] for r in rows if r[fld]]; b = [r["views"] for r in rows if not r[fld]]
    print(f"{fld:15} есть n={len(a):2} {k(st.median(a)) if a else '-':>6}   нет n={len(b):2} {k(st.median(b)) if b else '-':>6}")
# регрессия с контролем
n = len(rows)
def col(f): return np.array([float(r[f]) if r[f] is not None else np.nan for r in rows])
topics = ["manifesto", "devices", "essay", "own_product", "business_advice"]
C = np.column_stack([np.ones(n)] + [[1.0 if r["topic"] == t else 0.0 for r in rows] for t in topics]
                    + [[1.0 if r["period"] == p else 0.0 for r in rows] for p in ("2026H1", "2026H2")]
                    + [np.log(col("dur")), col("cta_code")])
y_v = np.log10(col("views")); y_c = np.log10(col("comments") / col("views") * 100 + 0.01); y_l = np.log10(col("likes") / col("views") * 100)
C2 = np.column_stack([C, y_v])
def coef(X, y): return np.linalg.lstsq(X, y, rcond=None)[0]
def test(x, y, Cm, perms=4000):
    m = ~np.isnan(x); X = np.column_stack([Cm[m], x[m]]); b = coef(X, y[m])[-1]
    null = np.array([coef(np.column_stack([Cm[m], rng.permutation(x[m])]), y[m])[-1] for _ in range(perms)])
    return b, (np.sum(np.abs(null) >= abs(b)) + 1) / (perms + 1)
def dummy(cat, val): return np.array([1.0 if r[cat] == val else 0.0 for r in rows])
def circ(step): return np.array([1.0 if step in r["circle"] else 0.0 for r in rows])
print("\n## С контролем (тема, период, длительность, код; для вовлечённости ещё log views). ** p<.05, * p<.1")
tests = [("template=tutorial", dummy("template","tutorial")), ("template=case_story", dummy("template","case_story")),
  ("template=review", dummy("template","review")), ("template=essay", dummy("template","essay")),
  ("conflict=none", dummy("conflict","none")), ("conflict=viewer_vs_world", dummy("conflict","viewer_vs_world")), ("conflict=author_vs_world", dummy("conflict","author_vs_world")),
  ("stakes:none", col("st_none")), ("stakes:money", col("st_money")), ("stakes:time", col("st_time")), ("stakes:knowledge", col("st_knowledge")),
  ("stakes:job", col("st_job")), ("n_stakes", col("n_stakes")),
  ("pov=viewer", dummy("pov","viewer")), ("pov=own", dummy("pov","own")), ("pov=third", dummy("pov","third")), ("in_medias_res", col("in_medias_res")),
  ("transformation", col("transformation")), ("cliffhanger_end", col("cliff")), ("has_climax", col("has_climax")), ("climax_pos", col("climax_pos")),
  ("has_resolution", col("has_resolution")), ("n_circle", col("n_circle")), ("circle:need", circ("need")), ("circle:take", circ("take")),
  ("circle:change", circ("change")), ("circle:return", circ("return")), ("twist_share", col("twist_share")), ("setup_share", col("setup_share")),
  ("step_share", col("step_share")), ("insight_share", col("insight_share")), ("has_promise", col("has_promise")), ("promise_pos", col("promise_pos")),
  ("second=filler", dummy("second","filler")), ("second=twist", dummy("second","twist")), ("second=step/demo/promise", np.array([1.0 if r["second"] in ("step","demo","promise") else 0.0 for r in rows])),
  ("open_loops", col("open_loops"))]
print(f"{'признак':26} {'views b / p':>16} {'comm/view b / p':>18} {'likes/view b / p':>18}")
for name, x in tests:
    if x[~np.isnan(x)].std() == 0: continue
    bv, pv = test(x, y_v, C); bc, pc = test(x, y_c, C2); bl, pl = test(x, y_l, C2)
    mk = lambda p: "**" if p < .05 else ("*" if p < .1 else "  ")
    print(f"{name:26} {bv:+.3f} {pv:.3f}{mk(pv)}   {bc:+.3f} {pc:.3f}{mk(pc)}   {bl:+.3f} {pl:.3f}{mk(pl)}")
json.dump([{**r, "circle": sorted(r["circle"])} for r in rows], open(RES / "story_rows_v4.json", "w"), ensure_ascii=False, indent=0)
