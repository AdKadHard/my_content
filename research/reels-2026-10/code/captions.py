"""Подписи (caption) к рилзам: вопрос / код в комменты / список инструментов / призыв поделиться / ничего
→ просмотры, комменты, лайки с контролем факторов. python3 code/captions.py → results/captions_<дата>.txt"""
import csv, re, datetime, collections, numpy as np
from _controls import D, RES, R, CODES, run_tests
S = {r["shortcode"]: r for r in csv.DictReader(open(D / "reels_stats_2026-10-12.csv"))}
DATE = datetime.date.today().isoformat(); OUT = RES / f"captions_{DATE}.txt"; open(OUT, "w").close()
def cat(c):
    c = c.strip(); cl = c.lower()
    if re.search(r"\d{3,}|пиши|напиши|в коммент", cl): return "code"
    if "?" in c: return "question"
    if re.search(r"поделись|перешли|отмечай|подпис", cl): return "share"
    if re.search(r"[a-z]{3,}", cl) and re.search(r"\+|,|промпт|сетап|девайс", cl): return "tools"
    return "other"
cats = {c: cat(S[c]["caption_head"]) for c in CODES}
lines = [f"# Подписи, {DATE}, n={len(CODES)}\n"]
for k in ["code", "question", "share", "tools", "other"]:
    mem = [c for c in CODES if cats[c] == k]
    if not mem: continue
    v = [float(r["views"]) for r in R if r["code"] in mem]; cr = [float(r["comment_rate"]) for r in R if r["code"] in mem]
    lr = [float(r["like_rate"]) for r in R if r["code"] in mem]
    lines.append(f"## {k:9} n={len(mem):2}  медиана просмотров {int(np.median(v))//1000}K  комменты/просмотр {np.median(cr):.2f}%  лайки/просмотр {np.median(lr):.2f}%")
    for c in sorted(mem, key=lambda c: -int(S[c]["plays"])): lines.append(f"   {c} {int(S[c]['plays'])//1000}K  {S[c]['caption_head']}")
open(OUT, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
feats = {f"cap_{k}": [int(cats[c] == k) for c in CODES] for k in ["code", "question", "share", "tools", "other"]}
feats["cap_len"] = [len(S[c]["caption_head"]) for c in CODES]
run_tests(feats, OUT, "\n## Проверка с контролем (внимание: cta_code в контроле уже включает код из подписи)")
