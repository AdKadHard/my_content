"""Трудоёмкость (ответы Аделя: быстро / средне / долго) против просмотров, лайков, комментов, формата, формы, темы.
Выход на единицу труда; доля хитов по трудоёмкости; какие дешёвые ролики летят. python3 code/effort.py → results/effort_<дата>.txt"""
import csv, json, glob, datetime, statistics as st, collections, re
import numpy as np
from _controls import D, A, RES, R, CODES, run_tests
S = {r["shortcode"]: r for r in csv.DictReader(open(D / "reels_stats_2026-10-12.csv"))}
L = {r["shortcode"]: r for r in csv.DictReader(open(D / "labels_2026-10-12.csv"))}
E = {json.loads(l)["code"]: json.loads(l)["effort"] for l in open(D / "effort_adel.jsonl")}
HOURS = {"fast": 0.5, "mid": 2, "long": 5}  # опорные: миллионный 4–5 ч, мелкие полчаса
FR = {json.load(open(f))["code"]: json.load(open(f)) for f in glob.glob(str(A / "frames_v1" / "*.json"))}
V4 = {c: json.load(open(A / "story_v4" / f"{c}.json"))["story"] for c in CODES}
EXCL = set(re.findall(r"`([A-Za-z0-9_-]{11})`", open(D / "EXCLUDED.md").read()))
ALL = [c for c in S if c in E and S[c]["date"] >= "2025-01-01" and c not in EXCL and S[c]["date"] <= "2026-10-01"]
hits = [c for c in ALL if int(S[c]["plays"]) >= 10000]
DATE = datetime.date.today().isoformat(); OUT = RES / f"effort_{DATE}.txt"; open(OUT, "w").close()
pl = lambda c: int(S[c]["plays"]); med = lambda cs: st.median(pl(c) for c in cs) / 1000 if cs else 0
out = [f"# Трудоёмкость, {DATE}. {len(ALL)} роликов 2025-01..2026-10-01 с ответом Аделя, из них хитов {len(hits)}. Часы условные: быстро 0.5, средне 2, долго 5\n"]
out.append("## 1. По трудоёмкости: n, доля хитов, медиана просмотров, просмотров на час, лайков и комментов на час, сумма просмотров")
for e in ("fast", "mid", "long"):
    v = [c for c in ALL if E[c] == e]; h = [c for c in v if c in hits]
    out.append(f"  {e:5} n={len(v):2}  хитов {len(h):2} ({len(h)/len(v)*100:.0f}%)  медиана {med(v):5.0f}K  медиана хитов {med(h):5.0f}K  "
               f"просм/час {st.median(pl(c)/HOURS[e] for c in v)/1000:5.0f}K  лайков/час {st.median(int(S[c]['likes'])/HOURS[e] for c in v):5.0f}  "
               f"комм/час {st.median(int(S[c]['comments'])/HOURS[e] for c in v):4.0f}  сумма {sum(pl(c) for c in v)/1e6:.1f}M за {sum(HOURS[e] for c in v):.0f} ч")
tot_h = sum(HOURS[E[c]] for c in ALL); tot_v = sum(pl(c) for c in ALL)
out.append(f"  всего {tot_v/1e6:.1f}M просмотров за ~{tot_h:.0f} ч; {tot_v/tot_h/1000:.0f}K на час в среднем")
out.append("\n## 2. Быстрые хиты (быстро и ≥50K): что у них общего")
for c in sorted([c for c in hits if E[c] == "fast" and pl(c) >= 50000], key=lambda c: -pl(c)):
    out.append(f"  {pl(c)//1000:5}K {FR.get(c,{}).get('format','?'):15} {L[c]['topic'] if c in L else '?':15} {V4[c]['template'] if c in V4 else '?':10} {V4[c]['pov'] if c in V4 else '?':7} «{(json.load(open(D/'transcripts_full_2026-10-12.json')).get(c,''))[:60]}»")
out.append("\n## 3. Долгие и средние, которые не окупились (средне/долго и <40K)")
for c in sorted([c for c in ALL if E[c] in ("mid", "long") and pl(c) < 40000], key=lambda c: -pl(c)):
    out.append(f"  {E[c]:4} {pl(c)//1000:5}K {FR.get(c,{}).get('format','?'):15} {L[c]['topic'] if c in L else 'провал':15} {V4[c]['template'] if c in V4 else '—':10} {S[c]['caption_head'][:40]}")
def table(name, key):
    out.append(f"\n## {name}: по строкам — категория, по колонкам — трудоёмкость: n / медиана просмотров")
    g = collections.defaultdict(list)
    for c in ALL: g[(key(c), E[c])].append(c)
    cats = sorted({key(c) for c in ALL}, key=lambda k: -med([c for c in ALL if key(c) == k]))
    out.append(f"  {'':16}" + "".join(f"{e:>14}" for e in ("fast", "mid", "long")) + "      все")
    for k in cats:
        out.append(f"  {str(k):16}" + "".join((f"{len(g[(k,e)]):2}/{med(g[(k,e)]):5.0f}K" if g[(k, e)] else "         —").rjust(14) for e in ("fast", "mid", "long")) + f"   {len([c for c in ALL if key(c)==k]):2}/{med([c for c in ALL if key(c)==k]):4.0f}K")
table("4. Формат по кадрам", lambda c: FR.get(c, {}).get("format", "?"))
table("5. Тема (хиты; провалы = «провал»)", lambda c: L[c]["topic"] if c in L else "провал")
table("6. Форма v4 (только хиты)", lambda c: V4[c]["template"] if c in V4 else "—")
table("7. Период", lambda c: L[c]["period"] if c in L else ("2026H2" if S[c]["date"] >= "2026-06-01" else "ранее"))
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out))
feats = {"трудоёмкость (1–3)": [{"fast": 1, "mid": 2, "long": 3}[E[c]] for c in CODES], "долго": [int(E[c] == "long") for c in CODES], "быстро": [int(E[c] == "fast") for c in CODES]}
run_tests(feats, OUT, "\n## 8. Хиты с контролем темы, периода, длительности, кода")
