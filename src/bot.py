# src/bot.py
import os
import re
import json
import logging
import sqlite3
import asyncio
from pathlib import Path

import numpy as np
from numpy.linalg import norm

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message
from aiogram.filters import CommandStart

from openai import OpenAI

# -----------------------------
# Configuration / Env
# -----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not set in environment")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

# Paths
BASE_DIR = Path(__file__).resolve().parent
LAW_BASE_PATH = BASE_DIR / "law_base.json"    # ensure this file exists (created by parse_law.py)
EMBED_PATH = BASE_DIR / "embeddings" / "tk_vectors.pkl"

# -----------------------------
# Init clients
# -----------------------------
bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Load law base and embeddings
# -----------------------------
if not LAW_BASE_PATH.exists():
    logging.warning(f"law_base.json not found at {LAW_BASE_PATH}; bot will run but article search unavailable.")
    LEX_BASE = {}
else:
    with open(LAW_BASE_PATH, "r", encoding="utf-8") as f:
        LEX_BASE = json.load(f)

VECTORS = {}
if EMBED_PATH.exists():
    try:
        import pickle
        with open(EMBED_PATH, "rb") as f:
            VECTORS = pickle.load(f)
    except Exception as e:
        logging.exception("Failed to load embeddings: %s", e)
        VECTORS = {}
else:
    VECTORS = {}

# -----------------------------
# Constants
# -----------------------------
LEX_LINK = "https://lex.uz/docs/6257291"
UZ_KEYWORDS = [
    "modda", "mehnat", "ishchi", "ish", "mehnat kodeksi", "qonun", "ishdan", "ish joyi",
    "ta'til", "maosh", "oylik", "kompensats", "bekor", "ish vaqti", "shartnoma", "salom"
]
RU_KEYWORDS = [
    "статья", "труд", "работник", "увольн", "отпуск", "больнич", "договор", "оклад",
    "компенсац", "сокращ", "дисциплинар", "прекращени", "приём", "работа"
]

# -----------------------------
# Utilities
# -----------------------------
def cosine(a, b):
    a = np.array(a); b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def detect_language_by_text(text: str):
    """
    Heuristic language detection:
    - if text contains Cyrillic letters -> 'ru'
    - if text contains Latin letters and Uzbek keywords -> 'uz'
    - otherwise -> None (unknown)
    """
    s = text.strip()
    if not s:
        return None
    # Cyrillic detection
    if re.search(r"[А-Яа-яЁё]", s):
        return "ru"
    # Uzbek keywords (latin)
    low = s.lower()
    for kw in UZ_KEYWORDS:
        if kw in low:
            return "uz"
    # quick check: if text contains any Latin letters and some uz tokens
    if re.search(r"[A-Za-z]", s):
        if any(k in low for k in ["ish", "modda", "mehnat", "qonun", "salom"]):
            return "uz"
    return None

# -----------------------------
# Simple state (in-memory)
# -----------------------------
# stores user -> language ("ru"/"uz")
USER_LANG = {}

# -----------------------------
# Commands
# -----------------------------
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! 👋 Send me a question about labour/HR or just write 'salom' to identify your language.\n\n"
        "Salom! 👋 Mehnat/HR borasidagi savolingizni yozing yoki tilni aniqlash uchun 'salom' deb yozing."
    )

@dp.message(F.text & F.regex(r"^(ru|uz)\b", flags=re.IGNORECASE))
async def set_lang_command(message: Message):
    code = message.text.strip().lower().split()[0]
    if code in ("ru", "uz"):
        USER_LANG[message.from_user.id] = code
        if code == "ru":
            await message.answer("Язык сохранён: Русский 🇷🇺")
        else:
            await message.answer("Til saqlandi: Oʻzbekcha (latin) 🇺🇿")
    else:
        await message.answer("Send 'ru' or 'uz' to choose language.")

# -----------------------------
# Classifier: is this HR question?
# -----------------------------
async def classify_hr(question: str) -> str:
    try:
        prompt = (
            "Кратко классифицируй текст: HR или NOT_HR. HR — вопросы по Трудовому кодексу/трудовым отношениям, "
            "например: увольнение, отпуск, больничный, оплата, дисциплина, сокращение, трудовой договор и т.п. "
            "NOT_HR — приветствия, эмоции, личные сообщения, нейтральные фразы.\n\n"
            f"Текст: {question}\n\nОтвечай строго 'HR' или 'NOT_HR'."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты — бинарный классификатор. Отвечай строго 'HR' или 'NOT_HR'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=6
        )
        out = resp.choices[0].message.content.strip().upper()
        if out.startswith("HR"):
            return "HR"
        return "NOT_HR"
    except Exception:
        low = question.lower()
        if any(k in low for k in ["увол", "отпуск", "больнич", "договор", "труд", "работник", "modda", "ish", "mehnat"]):
            return "HR"
        return "NOT_HR"

# -----------------------------
# Helpers: extract explicit article number
# -----------------------------
def extract_explicit_article(text: str):
    low = text.lower()
    m1 = re.search(r"стат(?:ья|и)?\s*(\d{1,4})", low)
    if m1:
        return m1.group(1)
    m2 = re.search(r"(\d{1,4})\s*-\s*modda", low) or re.search(r"(\d{1,4})\s+modda\b", low)
    if m2:
        return m2.group(1)
    m3 = re.search(r"modda\s*(\d{1,4})", low)
    if m3:
        return m3.group(1)
    return None

# -----------------------------
# Core message handler
# -----------------------------
@dp.message(F.text)
async def handle_message(message: Message):
    uid = message.from_user.id
    text = (message.text or "").strip()
    if not text:
        await message.answer("Не понял сообщение — напишите, пожалуйста, вопрос по Трудовому кодексу или 'ru'/'uz' чтобы выбрать язык.")
        return

    lang = USER_LANG.get(uid)
    if not lang:
        lang = detect_language_by_text(text)

    if not lang:
        await message.answer(
            "Не смог определить язык. Пожалуйста, напишите 'ru' (для русского) или 'uz' (для узбекского, lotin) — затем повторите ваш вопрос.\n\n"
            "Tilni aniqlay olmadim. Iltimos, 'ru' yoki 'uz' deb yozing va so'ng savolingizni yuboring."
        )
        return

    USER_LANG[uid] = lang
    await message.chat.do("typing")

    explicit = extract_explicit_article(text)
    if explicit:
        article_id = explicit
    else:
        article_id = None

    classification = "HR"
    try:
        classification = await classify_hr(text) if not explicit else "HR"
    except Exception:
        classification = "HR"

    if classification == "NOT_HR":
        if lang == "ru":
            reply = ("Привет! 👋 У меня всё отлично — я HR-бот по Трудовому кодексу Узбекистана. "
                     "Если у вас есть вопрос по увольнению, отпуску, больничному, трудовым договорам или правам работников — спрашивайте.")
        else:
            reply = ("Salom! 👋 Hammasi joyida — men Oʻzbekiston Mehnat Kodeksi bo‘yicha HR-botman. "
                     "Agar ishdan boʻshatish, taʼtil, kasallik, mehnat shartnomalari yoki xodim huquqlari haqida savolingiz bo‘lsa — so‘rang.")
        await message.answer(reply + f"\n\nИсточник: {LEX_LINK}")
        return

    article_text = ""
    found_article = None

    if article_id:
        if str(article_id) in LEX_BASE:
            found_article = str(article_id)
            article_text = LEX_BASE[found_article].get(lang, "")
        else:
            found_article = None
            article_text = ""

    if not found_article and VECTORS:
        try:
            emb = client.embeddings.create(model="text-embedding-3-small", input=text)
            qvec = np.array(emb.data[0].embedding)
            best_id = None
            best_score = -999
            for aid, vec in VECTORS.items():
                sc = cosine(qvec, np.array(vec))
                if sc > best_score:
                    best_score = sc
                    best_id = aid
            if best_score is not None and best_score >= 0.23:
                found_article = best_id
                article_text = LEX_BASE.get(str(found_article), {}).get(lang, "") or LEX_BASE.get(str(found_article), {}).get("ru", "")
            else:
                found_article = None
                article_text = ""
        except Exception as e:
            logging.exception("Embeddings search failed: %s", e)
            found_article = None
            article_text = ""

    system_msg = (
        "Ты — HR-консультант по Трудовому кодексу Республики Узбекистан. "
        "Если тебе передали текст статьи — отвечай ИСКЛЮЧИТЕЛЬНО на основе этого текста и не называй другие номера статей. "
        "Если текст статьи не передан — честно скажи: 'Статья не найдена в локальной базе; рекомендую посмотреть на lex.uz' и не придумывай номер. "
        "Отвечай кратко, ясно и давай практические шаги для кадровика."
    )

    if found_article and article_text:
        user_msg = (
            f"Вопрос ({lang}): {text}\n\n"
            f"Найдена статья #{found_article} (локальная база). Используй только этот текст для ответа.\n\n"
            f"Текст статьи:\n{article_text}\n\n"
            "Дай краткое понятное объяснение и практические шаги для кадровика."
        )
    else:
        user_msg = (
            f"Вопрос ({lang}): {text}\n\n"
            "Статья не найдена в локальной базе. Не придумывай номеров статей. Дай общий практический совет и порекомендуй проверить актуальную редакцию на lex.uz."
        )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.15,
            max_tokens=700
        )
        gpt_answer = completion.choices[0].message.content.strip()
    except Exception as e:
        logging.exception("GPT completion error: %s", e)
        if found_article and article_text:
            gpt_answer = "Сервис временно недоступен. Ниже — найденная статья:\n\n" + article_text
        else:
            gpt_answer = "Сервис временно недоступен. Попробуйте позже."

    footer = f"\n\nИсточник и актуальная редакция: {LEX_LINK}"
    if found_article:
        header = f"Найдена статья: {found_article}\n\n"
        await message.answer(header + gpt_answer + footer)
    else:
        await message.answer(gpt_answer + footer)

    try:
        conn = sqlite3.connect(str(BASE_DIR / "bot_logs.db"))
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                lang TEXT,
                question TEXT,
                answer TEXT,
                article INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            INSERT INTO logs (user_id, username, lang, question, answer, article)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            uid,
            message.from_user.username or "",
            lang,
            text,
            (("Найдена статья: " + str(found_article) + "\n\n") if found_article else "") + gpt_answer,
            int(found_article) if found_article else None
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.exception("Failed to write log: %s", e)

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
