"""Провалы (<10K) против хитов: что отличает ролики, которые не полетели. Текст (whisper, local/outputs/transcripts_fails.json),
монтаж (local/outputs/video_features), просодия (local/outputs/prosody), длительность и темп.
Провал = <10K просмотров, дата 2025-01-01..2026-09-30 (свежие ещё не отлежались, 2022–2023 — другая эпоха).
python3 code/fails.py → results/fails_<дата>.txt"""
import csv, json, datetime, re, numpy as np
from _lexicon import text_features
from _controls import D, RES, _R
OUTD = _R / "local" / "outputs"
S = {r["shortcode"]: r for r in csv.DictReader(open(D / "reels_stats_2026-10-12.csv"))}
L = {r["shortcode"]: r for r in csv.DictReader(open(D / "labels_2026-10-12.csv"))}
T = json.load(open(D / "transcripts_full_2026-10-12.json"))
TF = json.load(open(OUTD / "transcripts_fails.json")) if (OUTD / "transcripts_fails.json").exists() else {}
EXCL = set(re.findall(r"`([A-Za-z0-9_-]{11})`", open(D / "EXCLUDED.md").read()))
hits = [c for c in L if c not in EXCL]
fails = [c for c in S if int(S[c]["plays"]) < 10000 and "2025-01-01" <= S[c]["date"] <= "2026-09-30" and c not in EXCL]
hits_h2 = [c for c in hits if L[c]["period"] == "2026H2"]
DATE = datetime.date.today().isoformat(); OUT = RES / f"fails_{DATE}.txt"
rng = np.random.default_rng(0)
def feats(c):
    text = T.get(c) or TF.get(c) or ""
    r = {"dur": float(S[c]["duration_s"]), "has_speech": int(len(text.split()) >= 5)}
    if r["has_speech"]:
        tf, _ = text_features(text); r.update(tf); r["wpm"] = tf["n_words"] / r["dur"] * 60
    vf = OUTD / "video_features" / f"{c}.json"
    if vf.exists():
        d = json.load(open(vf)); r.update(cuts_per_10s=d["cuts_per_10s"], first_cut_s=d["first_cut_s"] if d["first_cut_s"] is not None else d["dur"],
                                           cuts_first_5s=d["cuts_first_5s"], ocr_t0=int(bool(d["ocr"]["t0"])), ocr_any=int(any(d["ocr"].values())))
    pr = OUTD / "prosody" / f"{c}.json"
    if pr.exists():
        d = json.load(open(pr)); ps = d["per_sec"]
        r.update(rms_mean=np.mean([x["rms"] for x in ps]), rms_first3=np.mean([x["rms"] for x in ps[:3]]),
                 f0_range_mean=np.nanmean([x["f0_range"] or np.nan for x in ps]), pauses_per_min=d["n_pauses_300ms"] / d["dur"] * 60)
    return r
FH = {c: feats(c) for c in hits}; FF = {c: feats(c) for c in fails}
def perm_p(a, b, n=5000):
    a, b = np.array(a, float), np.array(b, float); obs = abs(np.median(a) - np.median(b)); allv = np.concatenate([a, b])
    cnt = sum(abs(np.median(p[:len(a)]) - np.median(p[len(a):])) >= obs for p in (rng.permutation(allv) for _ in range(n)))
    return (cnt + 1) / (n + 1)
lines = [f"# Провалы против хитов, {DATE}", f"провалов {len(fails)}, хитов {len(hits)} (из них 2026-H2: {len(hits_h2)})", ""]
lines.append("## Провалы: что это за ролики")
for c in sorted(fails, key=lambda c: -int(S[c]["plays"])):
    text = T.get(c) or TF.get(c) or ""
    lines.append(f"  {c} {int(S[c]['plays']):5}  {S[c]['date']} {float(S[c]['duration_s']):5.0f}s  подпись: «{S[c]['caption_head']}»  речь: {text[:110] or '—'}")
keys = ["dur", "has_speech", "n_words", "wpm", "money_share", "time_share", "people_share", "tools_per100", "n_tools_unique", "numbers_per100",
        "num_money", "abstract_share", "zipf_mean", "analogies_per100", "cuts_per_10s", "first_cut_s", "cuts_first_5s", "ocr_t0", "ocr_any",
        "rms_mean", "rms_first3", "f0_range_mean", "pauses_per_min"]
lines.append("\n## Медианы: провалы / хиты 2026-H2 / все хиты (p — перестановочный тест разницы медиан, провалы vs хиты 2026-H2)")
lines.append(f"{'признак':18} {'провалы':>9} {'хиты H2':>9} {'все хиты':>9}   p")
for k in keys:
    a = [FF[c][k] for c in fails if k in FF[c]]; b = [FH[c][k] for c in hits_h2 if k in FH[c]]; d = [FH[c][k] for c in hits if k in FH[c]]
    if len(a) < 3 or len(b) < 3: continue
    p = perm_p(a, b)
    lines.append(f"{k:18} {np.median(a):9.2f} {np.median(b):9.2f} {np.median(d):9.2f}   {p:.3f}{' **' if p < 0.05 else (' *' if p < 0.1 else '')}  (n={len(a)}/{len(b)})")
open(OUT, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
