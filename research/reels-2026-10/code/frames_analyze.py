"""Визуальный слой по разметке кадров (annotations/frames_v1): формат, лицо/экран/девайс/плашка в первых секундах
→ просмотры (все 83 с медианами, хиты с контролем), провалы против хитов. python3 code/frames_analyze.py → results/frames_<дата>.txt"""
import csv, json, glob, datetime, statistics as st, collections, re
import numpy as np
from _controls import D, A, RES, R, CODES, run_tests
S = {r["shortcode"]: r for r in csv.DictReader(open(D / "reels_stats_2026-10-12.csv"))}
L = {r["shortcode"]: r for r in csv.DictReader(open(D / "labels_2026-10-12.csv"))}
FR = {}
for f in glob.glob(str(A / "frames_v1" / "*.json")):
    d = json.load(open(f)); FR[d["code"]] = d
EXCL = set(re.findall(r"`([A-Za-z0-9_-]{11})`", open(D / "EXCLUDED.md").read()))
fails = [c for c in FR if int(S[c]["plays"]) < 10000 and "2025-01-01" <= S[c]["date"] <= "2026-09-30" and c not in EXCL]
hits = [c for c in FR if c in L and c not in EXCL]
DATE = datetime.date.today().isoformat(); OUT = RES / f"frames_{DATE}.txt"; open(OUT, "w").close()
out = [f"# Визуальный слой, {DATE}. Размечено {len(FR)} роликов, хитов {len(hits)}, провалов {len(fails)}\n"]
def med(cs): return st.median(int(S[c]["plays"]) for c in cs) / 1000
out.append("## 1. Формат (по кадрам) → n / медиана просмотров, хиты и все")
g = collections.defaultdict(list)
for c in FR: g[FR[c]["format"]].append(c)
for k, v in sorted(g.items(), key=lambda x: -med(x[1])):
    h = [c for c in v if c in hits]; out.append(f"  {k:16} все n={len(v):2} {med(v):6.0f}K   хиты n={len(h):2} {med(h) if h else 0:6.0f}K   провалов {sum(c in fails for c in v)}")
out.append("\n## 2. Тема × формат (хиты): n / медиана")
g = collections.defaultdict(list)
for c in hits: g[(L[c]["topic"], FR[c]["format"])].append(c)
for k, v in sorted(g.items(), key=lambda x: (x[0][0], -med(x[1]))):
    if len(v) >= 2: out.append(f"  {k[0]:16} {k[1]:16} n={len(v):2} {med(v):6.0f}K")
out.append("\n## 3. Первый кадр (t0) и первые 3 секунды: есть / нет → медиана просмотров (хиты)")
for key, label in (("face", "лицо в t0"), ("screen", "экран в t0"), ("device", "девайс в t0")):
    a = [c for c in hits if FR[c]["t0"][key]]; b = [c for c in hits if not FR[c]["t0"][key]]
    out.append(f"  {label:14} есть n={len(a):2} {med(a) if a else 0:6.0f}K   нет n={len(b):2} {med(b) if b else 0:6.0f}K")
a = [c for c in hits if FR[c]["t0"]["text"].strip()]; b = [c for c in hits if not FR[c]["t0"]["text"].strip()]
out.append(f"  {'плашка в t0':14} есть n={len(a):2} {med(a) if a else 0:6.0f}K   нет n={len(b):2} {med(b) if b else 0:6.0f}K")
out.append("\n## 4. Провалы против хитов 2026-H2 (медианы)")
h2 = [c for c in hits if L[c]["period"] == "2026H2"]
for key in ("face_frames", "screen_frames", "device_frames", "text_frames", "shot_changes_0_3"):
    out.append(f"  {key:18} провалы {st.median(FR[c][key] for c in fails):.1f}   хиты H2 {st.median(FR[c][key] for c in h2):.1f}   все хиты {st.median(FR[c][key] for c in hits):.1f}")
out.append("  форматы провалов: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(FR[c]["format"] for c in fails).most_common()))
out.append("  форматы хитов H2: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(FR[c]["format"] for c in h2).most_common()))
out.append("  место съёмки провалов: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(FR[c].get("setting", "?") for c in fails).most_common()))
out.append("  место съёмки хитов: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(FR[c].get("setting", "?") for c in hits).most_common()))
out.append("\n## 5. Плашки первого кадра (текст t0) у топ-15 и у провалов")
for c in sorted(hits, key=lambda c: -int(S[c]["plays"]))[:15]: out.append(f"  {int(S[c]['plays'])//1000:5}K {FR[c]['format']:15} «{FR[c]['t0']['text'][:60]}»  {FR[c]['t0'].get('scene','')[:60]}")
for c in sorted(fails, key=lambda c: -int(S[c]["plays"])): out.append(f"  {int(S[c]['plays'])//1000:5}K {FR[c]['format']:15} «{FR[c]['t0']['text'][:60]}»  {FR[c]['t0'].get('scene','')[:60]}  [провал]")
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out))
feats = {}
for key in ("face_frames", "screen_frames", "device_frames", "text_frames", "shot_changes_0_3"): feats[key] = [FR[c][key] if c in FR else np.nan for c in CODES]
for key in ("face", "screen", "device"): feats[f"t0_{key}"] = [FR[c]["t0"][key] if c in FR else np.nan for c in CODES]
feats["t0_text"] = [int(bool(FR[c]["t0"]["text"].strip())) if c in FR else np.nan for c in CODES]
for fmt in ("talking_head", "talk_screen", "device_review", "cinematic_promo"): feats[f"fmt={fmt}"] = [int(FR[c]["format"] == fmt) if c in FR else np.nan for c in CODES]
for k, v in feats.items():
    v = np.array(v, float); feats[k] = np.where(np.isnan(v), np.nanmedian(v), v)
run_tests(feats, OUT, "\n## 6. Хиты с контролем факторов")
