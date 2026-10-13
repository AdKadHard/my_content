"""Повторы и перепосты: ролики с тем же хуком или близким текстом (TF-IDF cos ≥ 0.45), порядок по дате, просмотры по номеру повтора.
python3 code/reposts.py → results/reposts_<дата>.txt"""
import csv, json, datetime, re, statistics as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from _lexicon import text_features
D = __import__("_controls").D; RES = __import__("_controls").RES
S = {r["shortcode"]: r for r in csv.DictReader(open(D / "reels_stats_2026-10-12.csv"))}
T = json.load(open(D / "transcripts_full_2026-10-12.json"))
TF = json.load(open(D.parent / "local" / "outputs" / "transcripts_fails.json"))
ALL = {**TF, **T}; codes = sorted(ALL, key=lambda c: S[c]["date"])
docs = [text_features(ALL[c])[1] for c in codes]
M = TfidfVectorizer(min_df=1, sublinear_tf=True).fit_transform(docs); M = M / (np.sqrt(M.multiply(M).sum(1)) + 1e-9); sim = (M @ M.T).toarray()
first6 = {c: " ".join(re.findall(r"[а-яё]+", ALL[c].lower())[:6]) for c in codes}
DATE = datetime.date.today().isoformat(); out = [f"# Повторы и перепосты, {DATE}. {len(codes)} роликов с текстом\n"]
groups = []; used = set()
for i, c in enumerate(codes):
    if c in used: continue
    g = [c]
    for j in range(i + 1, len(codes)):
        d = codes[j]
        if d in used: continue
        if sim[i, j] >= 0.35 or first6[c] == first6[d]: g.append(d); used.add(d)
    if len(g) > 1: groups.append(g); used.add(c)
out.append("## Группы повторов (по дате): ролик, просмотры, похожесть на первый, первые слова")
decay = []
for g in groups:
    out.append("")
    for k, c in enumerate(g):
        v = int(S[c]["plays"]); i0 = codes.index(g[0]); ic = codes.index(c)
        out.append(f"  #{k+1} {c} {S[c]['date']} {v/1000:6.0f}K  sim {sim[i0, ic]:.2f}  «{ALL[c][:70]}»")
        if k > 0: decay.append((k + 1, v / max(int(S[g[0]]["plays"]), 1)))
out.append(f"\nГрупп повторов: {len(groups)}. Отношение просмотров повтора к первому ролику группы:")
for k in sorted({d[0] for d in decay}):
    v = [x for kk, x in decay if kk == k]; out.append(f"  повтор #{k}: n={len(v)} медиана ×{st.median(v):.2f}, min ×{min(v):.2f}, max ×{max(v):.2f}")
out.append("\n## Похожесть на ближайший предыдущий ролик (max cos за 5 предыдущих) → просмотры, только хиты с контролем: см. semantics (max_sim_recent5)")
near = []
for i, c in enumerate(codes):
    if i == 0: continue
    j = max(range(max(0, i - 5), i), key=lambda j: sim[i, j]); near.append((sim[i, j], int(S[c]["plays"]), c, codes[j]))
hi = [v for s_, v, *_ in near if s_ >= 0.2]; lo = [v for s_, v, *_ in near if s_ < 0.2]
out.append(f"  похож на недавний (cos≥0.2): n={len(hi)} медиана {st.median(hi)/1000:.0f}K;  не похож: n={len(lo)} медиана {st.median(lo)/1000:.0f}K  (без контроля темы)")
open(RES / f"reposts_{DATE}.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
