"""Лексика и разбор текста для семантических признаков (общее для semantics.py и fails.py)."""
import re, collections
import numpy as np, pymorphy3
from wordfreq import zipf_frequency
m = pymorphy3.MorphAnalyzer()
MONEY = {"рубль", "доллар", "бакс", "цена", "стоить", "зарплата", "заработать", "заработок", "деньги", "бесплатный",
         "бесплатно", "платить", "оплата", "тысяча", "миллион", "дорогой", "дёшево", "дешёвый", "дорого", "бюджет",
         "доход", "прибыль", "продать", "продажа", "купить", "подписка", "окупить", "расход", "стоимость", "чек", "выручка"}
TIME = {"час", "минута", "секунда", "день", "неделя", "месяц", "год", "вечер", "утро", "быстро", "долго", "сутки",
        "мгновенно", "срок", "время", "пока", "сразу", "за", "ночь", "лет", "неделями", "месяцами"} - {"за", "пока"}
PEOPLE = {"люди", "человек", "команда", "сотрудник", "клиент", "ребята", "нанимать", "подписчик", "новичок", "родитель",
          "коллега", "друг", "фрилансер", "разработчик", "дизайнер", "маркетолог", "предприниматель", "блогер", "эксперт",
          "ученик", "заказчик", "пользователь", "аудитория", "девушка", "парень", "жена", "гайз", "чувак"}
TOOLS = re.compile(r"\b(claude|clotco|клод|gpt|chatgpt|чатгпт|cursor|курсор|n8n|mac ?mini|мак ?мини|openclaw|openclo|опенкло|"
                   r"cardputer|hidock|plaud|obsidian|notion|telegram|телеграм|whisper|ollama|gemini|midjourney|supabase|directual|"
                   r"copilot|копайлот|codex|кодекс|raspberry|bambu|m5stack|elevenlabs|heygen|davinci|давинчи|figma|github|"
                   r"vercel|lovable|bolt|replit|deepseek|llama|mistral|openai|anthropic|perplexity|graphify|comfy|stable diffusion|"
                   r"iphone|айфон|ipad|macbook|макбук|linux|windows|docker|python|питон|js|react|sql|api|mcp|мсп|json|промпт|промт|"
                   r"clotco|клот|клод код|opencl\w*|опенкло\w*|perplexity|перплексити|обсидиан|notion|нотион|directual|директуал|"
                   r"hyperframes|гиперфреймс|виспер|кардпьютер|картпьютер|hidoc|хайдок|dji|osmo|распберри|бамбу|deepseek|дипсик|"
                   r"чат ?гпт|гпт|джемини|soloscout|clink|opng|upng|reddit|реддит|телега|тгк|сиды|spotify|zapier|make\.com|"
                   r"llm|ллм|нейронк\w+|нейросет\w+|агент\w*|бот\w*|сервер\w*|хостинг\w*|плат\w+|ардуино|arduino|esp32|пи\b)\b", re.I)
ANALOGY = re.compile(r"\b(будто|словно|похож\w*|это как\b|как если бы|такой же как|типа|представь\w*|по сути это)\b", re.I)
ABSTRACT = re.compile(r"(ость|ение|ание|ство|ция|зия|изм|тие|ота|изна)$")
NUM = re.compile(r"\d+[\d.,]*")
STOP = set("этот тот свой мой твой наш ваш быть весь такой который какой самый один два три очень просто вот это уже ещё еще только если как так что чтобы где когда потому тогда здесь там сейчас сегодня можно нужно надо делать сделать мочь хотеть давать дать есть нет да ну же бы ли или либо также тоже ещё более менее всё все вся всего кто куда зачем почему например типа блин".split())
WORD = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?")

def lemmas(text):
    out = []
    for w in WORD.findall(text.lower()):
        p = m.parse(w)[0]; out.append((p.normal_form, p.tag.POS))
    return out

def num_types(text):
    t = collections.Counter()
    for mo in NUM.finditer(text):
        s = mo.group(); after = text[mo.end(): mo.end() + 25].lower(); before = text[max(0, mo.start() - 12): mo.start()].lower()
        v = s.replace(",", ".").rstrip(".")
        if re.search(r"^\s*(%|процент)", after): t["percent"] += 1
        elif re.search(r"^\s*(долл|бакс|руб|тыс|к\b|k\b|\$|евро|миллион|млн)", after) or "$" in before: t["money"] += 1
        elif re.match(r"^(19|20)\d\d$", v) or re.search(r"^\s*(год|году|-м|-го)\b", after): t["year"] += 1
        elif re.search(r"^\s*(час|минут|секунд|сек|мин|дн|день|недел|месяц|лет\b|года\b)", after): t["time"] += 1
        else: t["count"] += 1
    return t


def text_features(text):
    """Признаки одного транскрипта: поля, абстрактность, редкость, числа, аналогии."""
    L = lemmas(text); n = len(L) or 1
    lem = [l for l, _ in L]; nouns = [l for l, pos in L if pos == "NOUN"]
    z = [zipf_frequency(l, "ru") for l in lem]
    nt = num_types(text); nn = sum(nt.values()); tools = len(TOOLS.findall(text))
    r = dict(n_words=n,
             money_share=sum(l in MONEY for l in lem) / n * 100,
             time_share=sum(l in TIME for l in lem) / n * 100,
             people_share=sum(l in PEOPLE for l in lem) / n * 100,
             tools_per100=tools / n * 100, n_tools_unique=len({x.lower() for x in TOOLS.findall(text)}),
             abstract_share=(sum(bool(ABSTRACT.search(l)) for l in nouns) / (len(nouns) or 1)) * 100,
             zipf_mean=float(np.mean(z)) if z else 0, rare_share=sum(x < 3.5 for x in z) / n * 100,
             numbers_per100=nn / n * 100, num_money=nt["money"], num_time=nt["time"], num_year=nt["year"],
             num_percent=nt["percent"], num_count=nt["count"],
             analogies_per100=len(ANALOGY.findall(text)) / n * 100,
             noun_share=len(nouns) / n * 100, verb_share=sum(pos in ("VERB", "INFN") for _, pos in L) / n * 100)
    doc = " ".join(l for l, pos in L if pos not in ("PREP", "CONJ", "PRCL", "NPRO", "INTJ") and l not in STOP and len(l) > 2)
    return r, doc
