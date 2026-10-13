"""Подача по фразам: просодия (local/outputs/prosody per_sentence: длительность, темп, громкость, высота, диапазон)
против функции фразы, накала и механизма из разметки. Скачок высоты/громкости на повороте, хук против тела, топ vs низ.
python3 code/prosody_phrases.py → results/prosody_phrases_<дата>.txt"""
import csv, json, datetime, statistics as st, collections
import numpy as np
from _controls import A, RES, R, CODES, run_tests, _R
OUTD = _R / "local" / "outputs"
DATE = datetime.date.today().isoformat(); OUT = RES / f"prosody_phrases_{DATE}.txt"; open(OUT, "w").close()
views = {r["code"]: float(r["views"]) for r in R}
rank = sorted(CODES, key=lambda c: -views[c]); TOP = set(rank[:30])
rows = []
for c in CODES:
    pr = json.load(open(OUTD / "prosody" / f"{c}.json"))["per_sentence"]
    lab = {x["i"]: x for x in json.load(open(A / f"labels_{c}.json"))["sentences"]}
    ps = {x["i"]: x for x in pr if "wpm" in x}
    med_rms = st.median(x["rms"] for x in ps.values()); med_f0 = st.median(x["f0_med"] for x in ps.values() if x["f0_med"])
    for i, x in ps.items():
        l = lab.get(i)
        if not l or not x["f0_med"]: continue
        prev = ps.get(i - 1)
        rows.append(dict(code=c, i=i, top=int(c in TOP), function=l["function"], arousal=int(l["arousal"]), emotion=l["emotion"], mech=l.get("mechanism") or "",
                         dur=x["dur"], wpm=x["wpm"], rms_rel=x["rms"] / med_rms, f0_rel=x["f0_med"] / med_f0, f0_range=x["f0_range"] or 0,
                         d_rms=(x["rms"] - prev["rms"]) / med_rms if prev else None,
                         d_f0=(x["f0_med"] - prev["f0_med"]) / med_f0 if prev and prev["f0_med"] else None,
                         d_wpm=x["wpm"] - prev["wpm"] if prev else None))
out = [f"# Подача по фразам, {DATE}. {len(rows)} фраз с просодией и разметкой. Громкость и высота — относительно медианы ролика (1.0 = как обычно)\n"]
def table(key, order=None):
    g = collections.defaultdict(list)
    for r in rows: g[r[key]].append(r)
    keys = order or sorted(g, key=lambda k: -len(g[k]))
    out.append(f"{key:12} {'n':>4} {'темп':>6} {'громк':>6} {'высота':>7} {'диапаз':>7} {'Δгромк':>7} {'Δвысота':>8} {'Δтемп':>6}")
    for k in keys:
        v = g[k]
        if len(v) < 5: continue
        f = lambda kk: st.mean(r[kk] for r in v if r[kk] is not None)
        out.append(f"{str(k):12} {len(v):4} {f('wpm'):6.0f} {f('rms_rel'):6.2f} {f('f0_rel'):7.2f} {f('f0_range'):7.0f} {f('d_rms'):+7.2f} {f('d_f0'):+8.2f} {f('d_wpm'):+6.0f}")
out.append("## 1. По функции фразы (Δ — скачок относительно предыдущей фразы)")
table("function", ["hook", "promise", "proof", "demo", "step", "insight", "twist", "payoff", "cta", "filler", "setup"])
out.append("\n## 2. По накалу из разметки (1–5)"); table("arousal", [1, 2, 3, 4, 5])
out.append("\n## 3. По механизму"); table("mech", ["contrast", "enemy", "identity", "self_disclosure", "humor", "authority", "reciprocity", "loss_aversion", "social_proof"])
out.append("\n## 4. Хук против тела, топ-30 против низ-30")
for grp, name in ((1, "топ-30"), (0, "низ-30")):
    h = [r for r in rows if r["top"] == grp and r["i"] == 0]; b = [r for r in rows if r["top"] == grp and r["i"] > 0]
    for nm, v in (("хук", h), ("тело", b)):
        out.append(f"{name} {nm:5} n={len(v):3} темп {st.mean(r['wpm'] for r in v):4.0f}  громк {st.mean(r['rms_rel'] for r in v):.2f}  высота {st.mean(r['f0_rel'] for r in v):.2f}  диапазон {st.mean(r['f0_range'] for r in v):3.0f}")
# совпадает ли подача с текстом: корреляция накала с просодией внутри ролика
out.append("\n## 5. Совпадение подачи и текста: корреляция (Спирмен) накала фразы с громкостью/высотой/темпом внутри ролика, медиана по роликам")
from scipy.stats import spearmanr
cor = collections.defaultdict(list)
for c in CODES:
    v = [r for r in rows if r["code"] == c]
    if len(v) < 6: continue
    for k in ("rms_rel", "f0_rel", "f0_range", "wpm"):
        rho = spearmanr([r["arousal"] for r in v], [r[k] for r in v]).correlation
        if rho == rho: cor[k].append((rho, c))
for k, v in cor.items():
    out.append(f"  накал × {k:9} медиана rho {st.median(x for x, _ in v):+.2f}; топ-30 {st.median(x for x, c in v if c in TOP):+.2f}, низ-30 {st.median(x for x, c in v if c not in TOP):+.2f}")
feats = {}
for c in CODES:
    v = [r for r in rows if r["code"] == c]
    if not v: continue
hook_feats = collections.defaultdict(dict)
for r in rows:
    if r["i"] == 0:
        hook_feats["темп хука"][r["code"]] = r["wpm"]; hook_feats["громкость хука (отн.)"][r["code"]] = r["rms_rel"]; hook_feats["высота хука (отн.)"][r["code"]] = r["f0_rel"]; hook_feats["диапазон хука"][r["code"]] = r["f0_range"]
for c in CODES:
    v = [r for r in rows if r["code"] == c]
    tw = [r for r in v if r["function"] == "twist" and r["d_f0"] is not None]
    hook_feats["скачок высоты на поворотах"][c] = st.mean(r["d_f0"] for r in tw) if tw else 0
    hook_feats["скачок громкости на поворотах"][c] = st.mean(r["d_rms"] for r in tw) if tw else 0
    hook_feats["согласие подачи с накалом (rho громк)"][c] = next((x for x, cc in cor["rms_rel"] if cc == c), 0)
    hook_feats["согласие подачи с накалом (rho высота)"][c] = next((x for x, cc in cor["f0_rel"] if cc == c), 0)
    hook_feats["разброс темпа по фразам (sd)"][c] = st.pstdev(r["wpm"] for r in v) if len(v) > 1 else 0
    hook_feats["разброс громкости (sd отн.)"][c] = st.pstdev(r["rms_rel"] for r in v) if len(v) > 1 else 0
for k, d in hook_feats.items():
    med = st.median(d.values()); feats[k] = [d.get(c, med) for c in CODES]
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out))
run_tests(feats, OUT, "\n## 6. Подача на уровне ролика → метрики (контроль факторов)")
