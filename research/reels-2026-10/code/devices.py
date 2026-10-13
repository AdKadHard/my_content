"""Речевые обороты и сценарные приёмы: где стоят в ролике, как связаны с накалом (arousal из разметки v2),
с функцией фразы и с метриками (контроль факторов). Пересечения тема × форма × хук.
python3 code/devices.py → results/devices_<дата>.txt, results/devices_phrases.csv"""
import csv, json, re, datetime, collections, statistics as st
import numpy as np, pymorphy3
from _controls import D, A, RES, R, CODES, run_tests, C, Y, coef, test
from _lexicon import TIME, MONEY
m = pymorphy3.MorphAnalyzer()
DATE = datetime.date.today().isoformat(); OUT = RES / f"devices_{DATE}.txt"; open(OUT, "w").close()
S = {r["shortcode"]: r for r in csv.DictReader(open(D / "reels_stats_2026-10-12.csv"))}
L = {r["shortcode"]: r for r in csv.DictReader(open(D / "labels_2026-10-12.csv"))}
V4 = {c: json.load(open(A / "story_v4" / f"{c}.json"))["story"] for c in CODES}
F = {r["code"]: r for r in R}
views = {c: float(F[c]["views"]) for c in CODES}
rank = sorted(CODES, key=lambda c: -views[c]); TOP = set(rank[:30]); BOT = set(rank[30:])

DEV = {
 "антитеза «не X, а Y»": re.compile(r"\bне\b[^.!?]{2,60}?,?\s+а\b", re.I),
 "вопрос": re.compile(r"\?"),
 "обращение «ты/вы»": re.compile(r"\b(ты|тебе|тебя|тобой|твой|твоя|твои|твоё|вы|вам|вас|ваш\w*)\b", re.I),
 "mock-quote «говорят/думал»": re.compile(r"\b(говорят|все говорят|считают|мол|думаешь|думал, что|думала|казалось|по идее|обещает)\b", re.I),
 "цифра": re.compile(r"\d"),
 "деньги": re.compile(r"\$|бакс\w*|доллар\w*|рубл\w*|тысяч\w*|\bk\b|профит|зарабат\w*|заработ\w*|цен[аыу]|стоит", re.I),
 "разговорное": re.compile(r"\b(блин|короче|го\b|чувак\w*|штуковин\w*|малышк\w*|фигн\w*|залета\w*|бабки|прикол\w*|жесть|норм\b|капец|ни фига|нафиг|офиген\w*|тупо\b|чел\b|пацан\w*|парень|ребят\w*|гайз|зачётн\w*|крут\w*|классн\w*|железк\w*|тачк\w*|видос\w*|рилс\w*|пилить|пилю|запилил\w*|ковыря\w*)\b", re.I),
 "пивот в начале фразы": re.compile(r"^(смотри\w*|по сути|давай\w*|короче|итого|в общем|ну и\b|кстати|а теперь|а вот|но вот|и вот|так вот|окей|ладно|го\b|поехали)", re.I),
 "интенсификатор": re.compile(r"\b(вообще|невероятн\w*|лют\w*|огромн\w*|жесточайш\w*|самый|самая|самое|идеальн\w*|лучш\w*|волшебн\w*|прекрасн\w*|зачаровыва\w*|сумасшедш\w*|бешен\w*|безумн\w*|реально|просто)\b", re.I),
 "первое лицо «я»": re.compile(r"\b(я|мне|меня|мной|мой|моя|мои|моё|у меня)\b", re.I),
 "дейксис «вот эта/вот так»": re.compile(r"\bвот (эт\w+|так\w*|здесь|тут|эти|такой|такая|такие|он|она|оно)\b", re.I),
 "самоирония/оговорка": re.compile(r"\b(естественно|конечно|разумеется|по-моему|мне кажется|как бы|не умею|не знаю|нулевы\w* знани\w*|не зная|не понимал|ошибался|дедушка|пылится)\b", re.I),
 "аналогия/метафора": re.compile(r"\b(будто|словно|как будто|это как|превращается в|похож\w*|это такой|та же|те же|типа)\b", re.I),
 "нумерация пунктов": re.compile(r"\b(первый|первое|второй|второе|третий|третье|во-первых|во-вторых|в-третьих|и наконец|наконец-то|четверт\w*|пят\w*)\b|^\d[.)]", re.I),
 "мат/грубость": re.compile(r"\b(бля\w*|нахуй|хуй\w*|суки|сука|пизд\w*|еба\w*|говно|жоп\w*|задниц\w*|насра\w*|отупл\w*|сдох\w*)\b", re.I),
 "тизер «в конце/узнаешь»": re.compile(r"\b(в конце|узнаешь|узнаете|чуть позже|отдельн\w+ видео|спойлер|пруф в конце|покажу, что получилось)\b", re.I),
 "обещание «покажу/расскажу»": re.compile(r"\b(покажу|расскажу|объясню|дам|пришлю|скину|разберём|разберемся|сэкономлю)\b", re.I),
 "императив": None,  # по морфологии
 "короткий удар (≤3 слов)": None,
 "сравнение цен (две цифры)": re.compile(r"\d[^.!?]*(бакс|доллар|\$)[^.!?]*\d[^.!?]*(бакс|доллар|\$)|(бакс|доллар|\$)[^.!?]*\d[^.!?]*\d", re.I),
}
def imperative(text):
    return any("impr" in str(m.parse(w)[0].tag) and "excl" in str(m.parse(w)[0].tag) for w in re.findall(r"[А-Яа-яЁё]+", text))

rows = []  # фразы
for c in CODES:
    sents = json.load(open(A / f"sentences_{c}.json"))["sentences"]; lab = json.load(open(A / f"labels_{c}.json"))["sentences"]
    n = len(sents)
    for s_, l in zip(sents, lab):
        t = s_["text"]; i = s_["i"]
        r = dict(code=c, i=i, n=n, rel=i / max(n - 1, 1), text=t, arousal=int(l["arousal"]), valence=int(l["valence"]), emotion=l["emotion"],
                 function=l["function"], mechanism=l.get("mechanism") or "", loop=l["loop"], you=int(l["you_focus"]), concrete=int(l["concrete"]),
                 nwords=len(t.split()), top=int(c in TOP))
        for k, rx in DEV.items():
            if k == "императив": r[k] = int(imperative(t))
            elif k == "короткий удар (≤3 слов)": r[k] = int(len(t.split()) <= 3)
            else: r[k] = int(bool(rx.search(t)))
        rows.append(r)
N = len(rows); keys = list(DEV)
with open(RES / "devices_phrases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
A_all = st.mean(r["arousal"] for r in rows)
def zone(rel): return "хук" if rel == 0 else ("начало" if rel < 0.34 else ("середина" if rel < 0.67 else "финал"))
out = [f"# Речевые обороты и сценарные приёмы, {DATE}. {len(CODES)} роликов, {N} фраз. Накал = arousal 1–5 из разметки v2 (средний {A_all:.2f})\n"]
out.append("## 1. Обороты: частота, где стоят, накал, функция, топ-30 vs низ-30")
out.append(f"{'оборот':30} {'фраз':>5} {'%':>5} | {'хук':>4} {'нач':>4} {'сер':>4} {'фин':>4} | {'накал':>5} {'Δ':>5} | {'топ%':>5} {'низ%':>5} | частые функции / эмоции")
for k in keys:
    sel = [r for r in rows if r[k]]
    if not sel: continue
    z = collections.Counter(zone(r["rel"]) for r in sel)
    ar = st.mean(r["arousal"] for r in sel)
    fn = collections.Counter(r["function"] for r in sel).most_common(2); em = collections.Counter(r["emotion"] for r in sel).most_common(2)
    tp = sum(r[k] for r in rows if r["top"]) / sum(r["top"] for r in rows) * 100; bt = sum(r[k] for r in rows if not r["top"]) / sum(1 - r["top"] for r in rows) * 100
    out.append(f"{k:30} {len(sel):5} {len(sel)/N*100:5.1f} | {z['хук']:4} {z['начало']:4} {z['середина']:4} {z['финал']:4} | {ar:5.2f} {ar-A_all:+5.2f} | {tp:5.1f} {bt:5.1f} | "
               + ", ".join(f"{a} {b}" for a, b in fn) + " / " + ", ".join(f"{a} {b}" for a, b in em))
out.append("\nΔ — накал фраз с оборотом минус средний. топ%/низ% — доля фраз с оборотом в 30 самых и 30 наименее просматриваемых роликах.")

out.append("\n## 2. Примеры: самые накалённые фразы с каждым оборотом (накал, функция, ролик, просмотры)")
for k in keys:
    sel = sorted([r for r in rows if r[k]], key=lambda r: (-r["arousal"], -views[r["code"]]))[:3]
    if not sel: continue
    out.append(f"\n**{k}**")
    for r in sel: out.append(f"  [{r['arousal']}] {r['function']:8} {r['code']} {views[r['code']]/1000:.0f}K  «{r['text'][:120]}»")

out.append("\n## 3. Накал по ходу ролика (средний arousal по пятым долям), топ-30 vs низ-30")
for grp, name in ((TOP, "топ-30"), (BOT, "низ-30")):
    bins = [[] for _ in range(5)]
    for r in rows:
        if r["code"] in grp: bins[min(int(r["rel"] * 5), 4)].append(r["arousal"])
    out.append(f"{name:7} " + "  ".join(f"{st.mean(b):.2f}" for b in bins) + f"   хук {st.mean(r['arousal'] for r in rows if r['code'] in grp and r['i']==0):.2f}")
out.append("\nЧто стоит в пиках (накал ≥4) чаще, чем в остальных фразах (lift = доля в пиках / доля в остальных):")
peak = [r for r in rows if r["arousal"] >= 4]; rest = [r for r in rows if r["arousal"] < 4]
lifts = []
for k in keys:
    a = sum(r[k] for r in peak) / len(peak); b = sum(r[k] for r in rest) / len(rest)
    if b > 0 and a > 0: lifts.append((a / b, k, a * 100, b * 100))
for lift, k, a, b in sorted(lifts, reverse=True): out.append(f"  {k:30} lift {lift:4.1f}  (в пиках {a:4.1f}%, в остальных {b:4.1f}%)")
out.append(f"\nФункции фраз в пиках: " + ", ".join(f"{a} {b}" for a, b in collections.Counter(r["function"] for r in peak).most_common(6)))
out.append(f"Механизмы в пиках: " + ", ".join(f"{a} {b}" for a, b in collections.Counter(r["mechanism"] for r in peak if r["mechanism"]).most_common(6)))
out.append("\nСпад после пика: что идёт следующей фразой после фразы с накалом ≥4 (функция):")
nxt = collections.Counter()
by = collections.defaultdict(dict)
for r in rows: by[r["code"]][r["i"]] = r
for r in peak:
    q = by[r["code"]].get(r["i"] + 1)
    if q: nxt[q["function"]] += 1
out.append("  " + ", ".join(f"{a} {b}" for a, b in nxt.most_common(8)))

out.append("\n## 4. Обороты в хуке (первая фраза): медиана просмотров есть / нет (n≥5)")
for k in keys:
    a = [views[r["code"]] for r in rows if r["i"] == 0 and r[k]]; b = [views[r["code"]] for r in rows if r["i"] == 0 and not r[k]]
    if len(a) >= 5 and len(b) >= 5: out.append(f"  {k:30} есть n={len(a):2} {st.median(a)/1000:5.0f}K   нет n={len(b):2} {st.median(b)/1000:5.0f}K")

# уровень ролика: доля фраз с оборотом → метрики с контролем
out.append("\n## 5. Доля фраз с оборотом в ролике → метрики (контроль: тема, период, длительность, код в комменты)")
feats = {}
for k in keys:
    feats[k] = [sum(r[k] for r in rows if r["code"] == c) / sum(1 for r in rows if r["code"] == c) * 100 for c in CODES]
# сценарные приёмы на уровне ролика
def reel(c):
    rs = sorted([r for r in rows if r["code"] == c], key=lambda r: r["i"]); n = len(rs); fn = [r["function"] for r in rs]
    opens = [r["i"] for r in rs if r["loop"] == "open"]; closes = [r["i"] for r in rs if r["loop"] == "close"]
    d = {}
    d["отложенное раскрытие"] = int(bool(opens) and opens[0] <= 1 and bool(closes) and closes[-1] / max(n - 1, 1) >= 0.6)
    d["поворот в первых 3 фразах"] = int("twist" in fn[:3])
    d["демо/шаг в первых 3 фразах"] = int(any(x in ("demo", "step") for x in fn[:3]))
    d["инсайт в финальной трети"] = int("insight" in fn[int(n * 0.67):])
    d["пик накала в хуке"] = int(rs[0]["arousal"] == max(r["arousal"] for r in rs))
    d["доля пиков (накал≥4)"] = sum(r["arousal"] >= 4 for r in rs) / n * 100
    d["размах накала (max−min)"] = max(r["arousal"] for r in rs) - min(r["arousal"] for r in rs)
    d["накал хука"] = rs[0]["arousal"]; d["средний накал"] = st.mean(r["arousal"] for r in rs)
    d["накал финала (последние 3)"] = st.mean(r["arousal"] for r in rs[-3:])
    d["смен функции на 10 фраз"] = sum(fn[i] != fn[i - 1] for i in range(1, n)) / n * 10
    d["доля «я»-фраз"] = sum(r["первое лицо «я»"] for r in rs) / n * 100; d["доля «ты»-фраз"] = sum(r["обращение «ты/вы»"] for r in rs) / n * 100
    d["«ты» минус «я»"] = d["доля «ты»-фраз"] - d["доля «я»-фраз"]
    last = " ".join(r["text"] for r in rs[-2:]).lower()
    d["CTA: код в комменты"] = int(bool(re.search(r"\d{3,}|напиш\w+ .{0,30}коммент|в коммент", last)))
    d["CTA: «до связи»"] = int("связи" in last); d["CTA: вопрос зрителю"] = int("?" in last)
    return d
RD = {c: reel(c) for c in CODES}
for k in RD[CODES[0]]: feats[k] = [RD[c][k] for c in CODES]
txt = run_tests(feats, OUT, "")
out.append(txt)

out.append("\n## 6. Тема × форма (v4): n / медиана просмотров")
tab = collections.defaultdict(list)
for c in CODES: tab[(F[c]["topic"], V4[c]["template"])].append(views[c])
topics = sorted({F[c]["topic"] for c in CODES}); temps = ["tutorial", "case_story", "review", "essay", "news_react", "other"]
out.append(f"{'тема':16}" + "".join(f"{t:>16}" for t in temps))
for tp in topics:
    out.append(f"{tp:16}" + "".join((f"{len(tab[(tp,t)]):2}/{st.median(tab[(tp,t)])/1000:4.0f}K" if tab[(tp, t)] else "       —").rjust(16) for t in temps))
out.append("\n## 7. Архетип хука (ручная разметка) × форма (v4): n / медиана просмотров")
tab = collections.defaultdict(list)
for c in CODES: tab[(L[c]["hook"], V4[c]["template"])].append(views[c])
hooks = sorted({L[c]["hook"] for c in CODES})
out.append(f"{'хук':16}" + "".join(f"{t:>16}" for t in temps))
for h in hooks:
    out.append(f"{h:16}" + "".join((f"{len(tab[(h,t)]):2}/{st.median(tab[(h,t)])/1000:4.0f}K" if tab[(h, t)] else "       —").rjust(16) for t in temps))
out.append("\n## 8. Форма × спор × призма (v4): n / медиана просмотров / медиана комментов на 1K")
tab = collections.defaultdict(list)
for c in CODES: tab[(V4[c]["template"], V4[c]["conflict"], V4[c]["pov"])].append(c)
for k, v in sorted(tab.items(), key=lambda x: -st.median(views[c] for c in x[1])):
    if len(v) >= 2: out.append(f"  {k[0]:11} {k[1]:16} {k[2]:7} n={len(v):2}  {st.median(views[c] for c in v)/1000:5.0f}K  комм/1K {st.median(float(S[c]['comments'])/views[c]*1000 for c in v):5.1f}")
out.append("\n## 9. Последовательности функций: самые частые тройки в топ-30 и низ-30 (на ролик)")
for grp, name in ((TOP, "топ-30"), (BOT, "низ-30")):
    tri = collections.Counter()
    for c in grp:
        fn = [r["function"] for r in sorted([r for r in rows if r["code"] == c], key=lambda r: r["i"])]
        seq = [fn[0]] + [fn[i] for i in range(1, len(fn)) if fn[i] != fn[i - 1]]
        for t in {tuple(seq[i:i + 3]) for i in range(len(seq) - 2)}: tri[t] += 1
    out.append(f"{name}: " + "; ".join(f"{'>'.join(t)} ({n})" for t, n in tri.most_common(8)))
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out))
