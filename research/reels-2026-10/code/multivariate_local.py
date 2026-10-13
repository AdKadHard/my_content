"""Локальные признаки (таймкоды, просодия, эмоция голоса, склейки, OCR, эмбеддинги) с контролем факторов.
Нужен local/outputs/features_local.csv (шаг 08). Пропуски заполняются медианой по столбцу (число пропусков печатается).
python3 code/multivariate_local.py → results/multivariate_local_<дата>.txt"""
import csv, datetime, numpy as np
from _controls import RES, CODES, run_tests, _R
F = {r["code"]: r for r in csv.DictReader(open(_R / "local" / "outputs" / "features_local.csv"))}
DATE = datetime.date.today().isoformat(); OUT = RES / f"multivariate_local_{DATE}.txt"; open(OUT, "w").close()
GROUPS = {
 "таймкоды (реальные секунды)": ["speech_start_s", "hook_end_s", "t_promise_real", "t_step_real", "t_payoff_real"],
 "просодия": ["rms_mean", "rms_first3", "f0_range_mean", "pauses_per_min", "wpm_hook", "wpm_mean"],
 "монтаж и плашки": ["n_cuts", "cuts_per_10s", "first_cut_s", "cuts_first_5s", "ocr_t0", "ocr_any"],
 "эмбеддинги e5": ["novelty_all", "max_sim_recent5", "sim_to_top10", "hook_body_sim"],
}
voice = sorted({k for r in F.values() for k in r if k.startswith("voice_") and k != "voice_hook_top"})
if voice: GROUPS["эмоция голоса (RESD, среднее по фразам)"] = voice
feats = {}; notes = []
for g, ks in GROUPS.items():
    for k in ks:
        if k not in next(iter(F.values())): notes.append(f"{k}: нет в features_local.csv"); continue
        raw = [F[c].get(k, "") for c in CODES]
        vals = np.array([float(x) if x not in ("", "None", "nan") else np.nan for x in raw])
        miss = int(np.isnan(vals).sum())
        if miss: notes.append(f"{k}: пропусков {miss}, заполнены медианой"); vals = np.where(np.isnan(vals), np.nanmedian(vals), vals)
        if k.startswith("t_"): vals = np.minimum(vals, 120)  # «нет такой фразы» → потолок
        feats[k] = vals
hdr = f"# Локальные признаки с контролем факторов, {DATE}\n" + "\n".join(notes)
open(OUT, "w").write(hdr + "\n"); print(hdr)
run_tests(feats, OUT, "\n## Проверка с контролем (тема, период, длительность, код в комменты)")
