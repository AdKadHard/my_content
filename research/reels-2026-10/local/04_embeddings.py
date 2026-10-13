"""Семантика: эмбеддинги фраз и роликов (multilingual-e5), новизна относительно предыдущих роликов,
близость к топ-10 по просмотрам, кластеры. Выход: local/embeddings.csv, local/clusters.txt, local/reel_vectors.npy"""
import json, csv, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from _common import *
model = SentenceTransformer("intfloat/multilingual-e5-base")
stats = {r["shortcode"]: r for r in csv.DictReader(open(DATA / "reels_stats_2026-10-12.csv"))}
T = json.load(open(DATA / "transcripts_full_2026-10-12.json"))
codes = [c for c in codes_hits() if c in T]
vec = {c: model.encode("passage: " + T[c], normalize_embeddings=True) for c in codes}
hook = {c: model.encode("passage: " + sentences(c)[0]["text"], normalize_embeddings=True) for c in codes}
order = sorted(codes, key=lambda c: stats[c]["date"])
top10 = sorted(codes, key=lambda c: -int(stats[c]["plays"]))[:10]
M = np.stack([vec[c] for c in codes]); np.save(OUT / "reel_vectors.npy", M)
labels = AgglomerativeClustering(n_clusters=7, metric="cosine", linkage="average").fit_predict(M)
rows = []
for k, c in enumerate(codes):
    prev = [vec[p] for p in order[:order.index(c)]]
    novelty = 1 - max((float(vec[c] @ p) for p in prev), default=0)
    recent = [vec[p] for p in order[max(0, order.index(c) - 5):order.index(c)]]
    sim_recent = max((float(vec[c] @ p) for p in recent), default=0)
    sim_top = np.mean([float(vec[c] @ vec[t]) for t in top10 if t != c])
    hook_body = float(hook[c] @ vec[c])
    rows.append(dict(code=c, cluster=int(labels[k]), novelty_all=round(novelty, 3), max_sim_recent5=round(sim_recent, 3),
                     sim_to_top10=round(sim_top, 3), hook_body_sim=round(hook_body, 3)))
with open(OUT / "embeddings.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
with open(OUT / "clusters.txt", "w") as f:
    for cl in range(7):
        mem = [c for k, c in enumerate(codes) if labels[k] == cl]
        f.write(f"\n## cluster {cl} n={len(mem)}\n")
        for c in sorted(mem, key=lambda c: -int(stats[c]["plays"])): f.write(f"  {c} {int(stats[c]['plays'])//1000}K  {sentences(c)[0]['text'][:80]}\n")
print("ok", len(rows))
