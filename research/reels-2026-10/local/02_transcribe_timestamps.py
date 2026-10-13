"""Пословные таймкоды через faster-whisper, затем привязка к фразам из emotion/sentences_<code>.json.
Выход: local/timestamps/<code>.json = {"words":[{w,start,end}], "sentences":[{i,start,end}]}"""
import json, re, difflib
from faster_whisper import WhisperModel
import soundfile as sf  # wav читаем сами: PyAV новых версий несовместим с faster-whisper
from _common import *
T = ensure(OUT / "timestamps")
model = WhisperModel("large-v3", compute_type="auto")  # на CPU без GPU замени на "small"
norm = lambda w: re.sub(r"[^\w]", "", w.lower())
for code in codes_hits():
    out = T / f"{code}.json"
    if out.exists(): continue
    wav = OUT / "audio" / f"{code}.wav"
    if not wav.exists(): print("no audio", code); continue
    segs, _ = model.transcribe(sf.read(str(wav), dtype="float32")[0], language="ru", word_timestamps=True, vad_filter=True)
    words = [{"w": w.word.strip(), "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]
    # выравнивание фраз Apify-транскрипта на слова whisper
    sents = sentences(code); ww = [norm(x["w"]) for x in words]
    res = []; pos = 0
    for s in sents:
        toks = [norm(t) for t in s["text"].split() if norm(t)]
        if not toks or pos >= len(words): res.append({"i": s["i"], "start": None, "end": None}); continue
        window = ww[pos:pos + len(toks) * 3 + 5]
        sm = difflib.SequenceMatcher(None, toks, window); blocks = [b for b in sm.get_matching_blocks() if b.size]
        if not blocks: res.append({"i": s["i"], "start": None, "end": None}); continue
        a = pos + blocks[0].b; b = pos + blocks[-1].b + blocks[-1].size - 1
        res.append({"i": s["i"], "start": words[a]["start"], "end": words[b]["end"]}); pos = b + 1
    json.dump({"words": words, "sentences": res}, open(out, "w"), ensure_ascii=False, indent=0)
    print("ok", code, len(words), "words")
