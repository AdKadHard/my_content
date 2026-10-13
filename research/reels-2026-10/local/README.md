# Локальный пайплайн: то, что не сделать из облачной сессии

Облачное окружение не пускает на `huggingface.co`, `*.cdninstagram.com`, `*.fbcdn.net`. Поэтому видео, голос, кадры и нейросетевые модели недоступны. Этот каталог — всё, что нужно запустить локально. Результаты пишутся в `data/instagram/local/`, их надо закоммитить, дальше анализ продолжается в облаке.

Скрипты написаны без возможности прогнать их здесь. Ожидай мелких правок при первом запуске (версии библиотек, имена полей моделей).

## Что это разблокирует

| Шаг | Что даёт | Какие гипотезы проверяет |
|---|---|---|
| 01 видео | mp4 всех 83 рилзов | всё ниже |
| 02 таймкоды | настоящие секунды каждой фразы (сейчас — оценка по доле слов) | обещание в первые 15 с, тайминг первого шага, наложение на кривые удержания |
| 03 эмоции текста (HF) | второй, независимый разметчик эмоций | сверка с разметкой Claude v2 (совпадение с Аделем 10/12) |
| 04 эмбеддинги | семантическая близость роликов, новизна, кластеры | «повтор темы выгорает», «близость к хитам», семантические кластеры × просмотры |
| 05 просодия | громкость, высота, паузы, темп по секундам | расхождение текста и подачи, темп в хуке |
| 06 эмоции голоса (HF) | эмоция по голосу на каждую фразу | «спокойный голос на злом тексте», сверка с текстом |
| 07 кадры и склейки | частота склеек, первые 3 секунды покадрово, текст-плашки (OCR) | визуальный хук, плотность монтажа, 6 форматов из скилла |
| 08 сборка | один CSV на ролик, стыкуется с `data/instagram/emotion/features_v2.csv` | мультивариантный анализ с визуалом и голосом |

## Установка

```bash
cd local
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg должен быть в PATH (brew install ffmpeg / apt install ffmpeg)
```

Модели с Hugging Face скачаются при первом запуске (~3 ГБ суммарно). Для `faster-whisper` на Mac без CUDA хватит `small`/`medium`, на GPU — `large-v3`.

## Порядок

```bash
python 01_download_videos.py          # data/instagram/local/video/<code>.mp4 + audio/<code>.wav
python 02_transcribe_timestamps.py    # local/timestamps/<code>.json
python 03_text_emotion_hf.py          # local/text_emotion/<code>.json + agreement.txt
python 04_embeddings.py               # local/embeddings.csv, clusters.txt
python 05_audio_prosody.py            # local/prosody/<code>.json
python 06_speech_emotion.py           # local/speech_emotion/<code>.json
python 07_video_cuts_frames.py        # local/video_features/<code>.json, frames/<code>/t0..t3.jpg
python 08_merge_features.py           # local/features_local.csv
```

Каждый скрипт пропускает ролики, для которых выход уже есть, так что можно перезапускать.

## Что закоммитить

Всё из `data/instagram/local/` **кроме** `video/` и `audio/` (они в `.gitignore`). Кадры `frames/` — да, они маленькие.

## Кривые удержания

Отдельно: скриншоты кривых из Instagram Insights клади в `data/instagram/retention/<code>.png`. Я сниму точки с графика и наложу на таймкоды из шага 02.
