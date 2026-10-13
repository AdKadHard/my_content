"""Семантика текста рилзов: лексические поля (деньги, время, инструменты, люди), абстрактность,
редкость слов (wordfreq), типы чисел, аналогии; TF-IDF-кластеры и новизна ролика относительно предыдущих.
Проверка с контролем факторов как в multivariate.py. Нужны pymorphy3, wordfreq, scikit-learn.
python3 code/semantics.py → results/semantics_features.csv, results/semantics_<дата>.txt"""
import csv, json, re, datetime, collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from _controls import D, RES, R, CODES, run_tests
from _lexicon import text_features
T = json.load(open(D / "transcripts_full_2026-10-12.json"))
S = {r["shortcode"]: r for r in csv.DictReader(open(D / "reels_stats_2026-10-12.csv"))}
DATE = datetime.date.today().isoformat()
OUT = RES / f"semantics_{DATE}.txt"; open(OUT, "w").close()

rows = []
docs = []
for c in CODES:
    r, doc = text_features(T[c]); rows.append({"code": c, **r}); docs.append(doc)

# TF-IDF: кластеры и новизна
V = TfidfVectorizer(min_df=2, max_df=0.8, sublinear_tf=True).fit_transform(docs)
M = V.toarray(); M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
sim = M @ M.T
order = sorted(range(len(CODES)), key=lambda i: S[CODES[i]]["date"])
rank = {i: k for k, i in enumerate(order)}
top10 = sorted(range(len(CODES)), key=lambda i: -float(R[i]["views"]))[:10]
K = 6
labels = KMeans(n_clusters=K, n_init=20, random_state=0).fit_predict(M)  # векторы нормированы → косинус
for i, c in enumerate(CODES):
    prev = [order[k] for k in range(rank[i])]
    rows[i]["novelty_all"] = 1 - max((sim[i, p] for p in prev), default=0)
    rec = [order[k] for k in range(max(0, rank[i] - 5), rank[i])]
    rows[i]["max_sim_recent5"] = max((sim[i, p] for p in rec), default=0)
    rows[i]["sim_to_top10"] = float(np.mean([sim[i, t] for t in top10 if t != i]))
    rows[i]["cluster"] = int(labels[i])
with open(RES / "semantics_features.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# описание кластеров
feat_names = V.shape and TfidfVectorizer(min_df=2, max_df=0.8, sublinear_tf=True).fit(docs).get_feature_names_out()
lines = [f"# Семантика текста, {DATE}, n={len(CODES)}\n", "## TF-IDF-кластеры (k-means на нормированных векторах, k=6)"]
for cl in range(K):
    mem = [i for i in range(len(CODES)) if labels[i] == cl]
    cent = M[mem].mean(0); top = [feat_names[j] for j in np.argsort(-cent)[:8]]
    views = sorted(float(R[i]["views"]) for i in mem)
    lines.append(f"\n### cluster {cl}  n={len(mem)}  медиана просмотров {int(np.median(views))//1000}K  слова: {', '.join(top)}")
    for i in sorted(mem, key=lambda i: -float(R[i]["views"])):
        lines.append(f"  {CODES[i]} {int(float(R[i]['views']))//1000}K [{R[i]['topic']}] {T[CODES[i]][:70]}")
lines.append("\n## Медианы по признакам (верхняя половина по просмотрам vs нижняя)")
half = sorted(range(len(CODES)), key=lambda i: -float(R[i]["views"]))
hi, lo = half[:len(half) // 2], half[len(half) // 2:]
for k in [x for x in rows[0] if x not in ("code", "cluster")]:
    a = np.median([rows[i][k] for i in hi]); b = np.median([rows[i][k] for i in lo])
    lines.append(f"{k:20} топ-30: {a:7.2f}   низ-30: {b:7.2f}")
open(OUT, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
feats = {k: [r[k] for r in rows] for k in rows[0] if k not in ("code", "cluster", "n_words")}
for cl in range(K): feats[f"cluster_{cl}"] = [int(r["cluster"] == cl) for r in rows]
run_tests(feats, OUT, "\n## Проверка с контролем факторов")
