"""Скачать mp4 всех рилзов и вытащить wav 16 кГц.
Сначала yt-dlp по URL рилса (может понадобиться --cookies-from-browser для Instagram),
если не вышло — прямой videoUrl из выгрузки Apify (ссылки живут ~сутки; обнови выгрузку
через apify/instagram-reel-scraper и положи items JSON в data/instagram/local/apify_items.json)."""
import subprocess, json, sys
from _common import *
V = ensure(OUT / "video"); A = ensure(OUT / "audio")
apify = {}
p = OUT / "apify_items.json"
if p.exists():
    for it in json.load(open(p)): apify[it["shortCode"]] = it.get("videoUrl")
COOKIES = sys.argv[1:]  # напр. --cookies-from-browser chrome
for code in codes_all():
    mp4 = V / f"{code}.mp4"; wav = A / f"{code}.wav"
    if not mp4.exists():
        url = f"https://www.instagram.com/reel/{code}/"
        r = subprocess.run(["yt-dlp", *COOKIES, "-o", str(mp4), "-f", "mp4", url], capture_output=True)
        if r.returncode != 0 and apify.get(code):
            subprocess.run(["curl", "-sL", "-o", str(mp4), apify[code]])
        if not mp4.exists(): print("FAIL", code); continue
    if not wav.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-ac", "1", "-ar", "16000", str(wav)])
    print("ok", code)
