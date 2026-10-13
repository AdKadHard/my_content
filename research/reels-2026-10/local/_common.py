import json, csv, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # research/reels-2026-10
DATA = ROOT / "data"
OUT = ROOT / "local" / "outputs"
EMO = ROOT / "annotations"
def codes_all():
    return [r["shortcode"] for r in csv.DictReader(open(DATA / "reels_stats_2026-10-12.csv"))]
def codes_hits():
    return json.load(open(EMO / "_codes.json"))
def sentences(code):
    return json.load(open(EMO / f"sentences_{code}.json"))["sentences"]
def ensure(p): Path(p).mkdir(parents=True, exist_ok=True); return Path(p)
def done(path): return Path(path).exists()
