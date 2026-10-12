# my_content

Весь контент Аделя Кадырова: сценарии рилзов @harsh.times, посты для телеграм-канала «ИИ для реального мира», данные и ToV-скиллы.

## Структура

- `skills/tov-adel-reels/` — ToV-скилл для Instagram Reels (корпус рилзов + attribution-анализ)
- `skills/tov-adel-telegram/` — ToV-скилл для телеграм-канала
- `reels/scripts/` — сценарии рилзов
- `telegram/posts/` — посты для канала
- `data/instagram/` — сырые выгрузки статистики рилзов (Apify)

## Данные Instagram

- `data/instagram/reels_stats_YYYY-MM-DD.csv` — статистика всех рилзов на дату выгрузки (Apify `apify/instagram-reel-scraper`)
- `data/instagram/labels_YYYY-MM-DD.csv` — разметка рилзов >10K: тема, архетип хука, период
- `data/instagram/analyze.py` — медианы по факторам: `python3 analyze.py labels_YYYY-MM-DD.csv`
- `data/instagram/transcripts_full_YYYY-MM-DD.json` — полные транскрипты (заново не транскрибировать)
- `data/instagram/transcripts/` — хуки и выдержки
- `research/` — исследования методов анализа
