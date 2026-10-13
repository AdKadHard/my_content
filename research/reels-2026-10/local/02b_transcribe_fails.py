"""Транскрипция роликов, которых нет в data/transcripts_full_2026-10-12.json (провалы <10K), через faster-whisper.
Выход: local/outputs/transcripts_fails.json {code: text}, local/outputs/timestamps_fails/<code>.json (слова с таймкодами)."""
import json
from faster_whisper import WhisperModel
import soundfile as sf  # wav читаем сами: PyAV новых версий несовместим с faster-whisper
from _common import *
T = json.load(open(DATA / "transcripts_full_2026-10-12.json"))
out_json = OUT / "transcripts_fails.json"; TS = ensure(OUT / "timestamps_fails")
res = json.load(open(out_json)) if out_json.exists() else {}
model = WhisperModel("large-v3", compute_type="auto")
for code in codes_all():
    if code in T or code in res: continue
    wav = OUT / "audio" / f"{code}.wav"
    if not wav.exists(): print("no audio", code); continue
    segs, _ = model.transcribe(sf.read(str(wav), dtype="float32")[0], language="ru", word_timestamps=True, vad_filter=True)
    words = [{"w": w.word.strip(), "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]
    res[code] = " ".join(w["w"] for w in words)
    json.dump({"words": words}, open(TS / f"{code}.json", "w"), ensure_ascii=False, indent=0)
    json.dump(res, open(out_json, "w"), ensure_ascii=False, indent=1)
    print("ok", code, len(words), "words")
