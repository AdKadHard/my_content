"""Проверка признаков с контролем факторов: тема, период, длительность, код в комменты.
Каждый признак добавляется к контрольной модели по одному (n=61, больше признаков сразу не выдержит).
p — перестановочный тест (признак перемешивается 5000 раз). python3 multivariate.py"""
from pathlib import Path as _P
_R = _P(__file__).resolve().parent.parent
D, A, RES = _R / "data", _R / "annotations", _R / "results"
import csv, json, math, os
import numpy as np
EXCLUDE = set(filter(None, os.environ.get("REELS_EXCLUDE", "").split(",")))
rng = np.random.default_rng(0)
R = [r for r in csv.DictReader(open(RES / "features_v2.csv")) if r["code"] not in EXCLUDE]
arc = {r["code"]: r["arc"] for r in json.load(open(RES / "summary.json"))}
for r in R: r["arc_rise"] = int(arc.get(r["code"]) == "rise")
n = len(R)
def col(k): return np.array([float(r[k]) for r in R])
def dummies(k, levels): return np.column_stack([[1.0 if r[k] == l else 0.0 for r in R] for l in levels])
topics = ["manifesto", "devices", "essay", "own_product", "business_advice"]  # база: ai_tools + прочее
C = np.column_stack([np.ones(n), dummies("topic", topics), dummies("period", ["2026H1", "2026H2"]),
                     np.log(col("dur")), col("cta_code")])
Y = {"log_views": col("log_views"), "log_comments_per_view": np.log10(col("comment_rate") + 0.01),
     "log_likes_per_view": np.log10(col("like_rate"))}
def coef(X, y): return np.linalg.lstsq(X, y, rcond=None)[0]
def test(x, y, perms=5000):
    X = np.column_stack([C, x]); b = coef(X, y)[-1]
    null = np.array([coef(np.column_stack([C, rng.permutation(x)]), y)[-1] for _ in range(perms)])
    return b, (np.sum(np.abs(null) >= abs(b)) + 1) / (perms + 1)
def r2(X, y):
    res = y - X @ coef(X, y); return 1 - res.var() / y.var()
FEATS = {
 "эмоции/приёмы": ["hook_identity", "hook_enemy", "hook_contrast", "enemy", "concrete_share", "hook_concrete",
                   "open_loops", "neg_viewer_share", "arc_rise"],
 "структура": ["t_promise", "t_proof", "t_step", "t_payoff", "promise_before_proof", "has_payoff", "has_twist",
               "blocks_per_10s", "step_share", "insight_share"],
}
print(f"n={n}. Контроль: тема ({', '.join(topics)} vs ai_tools/прочее), период, log(длительность), код в комменты.")
for yname, y in Y.items():
    print(f"\n### {yname}   R² только контроля = {r2(C, y):.2f}")
    print("контроль: код в комменты b={:+.3f}".format(coef(C, y)[-1]))
    for grp, fs in FEATS.items():
        print(f"-- {grp}")
        for f in fs:
            b, p = test(col(f), y)
            mark = " **" if p < 0.05 else (" *" if p < 0.1 else "")
            print(f"{f:22} b={b:+.3f}  p={p:.3f}{mark}")
