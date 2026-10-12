"""Медианы просмотров по факторам для рилзов @harsh.times (>10K). Запуск: python3 analyze.py labels_YYYY-MM-DD.csv"""
import csv, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    r["views"] = int(r["views"]); r["dur"] = float(r["dur"])
    d = r["dur"]
    r["dur_b"] = "<30s" if d < 30 else "30-45s" if d < 45 else "45-60s" if d < 60 else "60-90s" if d < 90 else "90+s"
k = lambda v: f"{v/1000:.0f}K"
print(f"n={len(rows)}  median={k(st.median(r['views'] for r in rows))}  mean={k(st.mean(r['views'] for r in rows))}")
for f in ["topic", "hook", "dur_b", "period"]:
    print(f"\n## {f}")
    g = {}
    for r in rows: g.setdefault(r[f], []).append(r["views"])
    for key, v in sorted(g.items(), key=lambda x: -st.median(x[1])):
        print(f"{key:16} n={len(v):2}  median={k(st.median(v)):>6}  mean={k(st.mean(v)):>6}  min={k(min(v)):>5}  max={k(max(v)):>6}")
print("\n## topic x hook (n>=3)")
g = {}
for r in rows: g.setdefault((r["topic"], r["hook"]), []).append(r["views"])
for key, v in sorted(g.items(), key=lambda x: -st.median(x[1])):
    if len(v) >= 3: print(f"{key[0]:16} {key[1]:10} n={len(v)} median={k(st.median(v))}")
