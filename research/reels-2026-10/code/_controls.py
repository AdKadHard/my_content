"""Общая контрольная модель для проверок признаков (та же, что в multivariate.py):
тема, период, log(длительность), код в комменты; p — перестановочный тест."""
from pathlib import Path as _P
_R = _P(__file__).resolve().parent.parent
D, A, RES = _R / "data", _R / "annotations", _R / "results"
import csv, os, numpy as np
rng = np.random.default_rng(0)
EXCLUDE = set(filter(None, os.environ.get("REELS_EXCLUDE", "").split(",")))  # напр. закреплённые ролики
R = [r for r in csv.DictReader(open(RES / "features_v2.csv")) if r["code"] not in EXCLUDE]
CODES = [r["code"] for r in R]
n = len(R)
def col(k): return np.array([float(r[k]) for r in R])
def dummies(k, levels): return np.column_stack([[1.0 if r[k] == l else 0.0 for r in R] for l in levels])
TOPICS = ["manifesto", "devices", "essay", "own_product", "business_advice"]
C = np.column_stack([np.ones(n), dummies("topic", TOPICS), dummies("period", ["2026H1", "2026H2"]),
                     np.log(col("dur")), col("cta_code")])
Y = {"log_views": col("log_views"), "log_comments_per_view": np.log10(col("comment_rate") + 0.01),
     "log_likes_per_view": np.log10(col("like_rate"))}
def coef(X, y): return np.linalg.lstsq(X, y, rcond=None)[0]
def test(x, y, perms=5000):
    x = np.asarray(x, float)
    if x.std() == 0: return 0.0, 1.0
    X = np.column_stack([C, x]); b = coef(X, y)[-1]
    null = np.array([coef(np.column_stack([C, rng.permutation(x)]), y)[-1] for _ in range(perms)])
    return b, (np.sum(np.abs(null) >= abs(b)) + 1) / (perms + 1)
def run_tests(feats, out, title=""):
    """feats: {name: vector по порядку CODES}. Пишет таблицу в файл out и печатает."""
    lines = [title, f"n={n}. Контроль: тема, период, log(длительность), код в комменты. p — перестановочный (5000)."]
    for yname, y in Y.items():
        lines.append(f"\n### {yname}")
        for f, x in feats.items():
            b, p = test(x, y)
            mark = " **" if p < 0.05 else (" *" if p < 0.1 else "")
            lines.append(f"{f:24} b={b:+.3f}  p={p:.3f}{mark}")
    txt = "\n".join(lines); print(txt); open(out, "a").write(txt + "\n"); return txt
